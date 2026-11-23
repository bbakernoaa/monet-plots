# src/monet_plots/plots/facet_grid.py
from __future__ import annotations

from itertools import cycle, islice, product
from math import ceil
from typing import Any, Callable

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.patches import Patch
import pandas as pd
import numpy as np
import xarray as xr

from ..plot_utils import to_dataframe
from ..style import set_style
from .base import BasePlot


def _get_grid_axes(grid):
    """Return the axes array from a FacetGrid, preferring .axs over deprecated .axes."""
    axes = getattr(grid, "axs", None)
    if axes is None:
        axes = getattr(grid, "axes", None)
    return axes


def _facet_levels(data, column, order):
    if column is None:
        return []
    if order is not None:
        return list(order)
    values = data[column]
    if isinstance(values.dtype, pd.CategoricalDtype):
        return list(values.cat.categories)
    levels = list(values.dropna().unique())
    return sorted(levels) if pd.api.types.is_numeric_dtype(values) else levels


def _facet_colors(palette, levels):
    if isinstance(palette, dict):
        return [palette[level] for level in levels]
    if isinstance(palette, str):
        # Matplotlib bundles the six-color palettes used by older seaborn styles.
        legacy = plt.style.library.get(f"seaborn-v0_8-{palette.removesuffix('6')}", {})
        if "axes.prop_cycle" in legacy:
            palette = legacy["axes.prop_cycle"].by_key()["color"]
        else:
            cmap = colormaps["Dark2" if palette in {"dark", "dark6"} else palette]
            if cmap.N > 20:
                return list(cmap(np.linspace(0, 1, len(levels) + 2)[1:-1]))
            palette = cmap.colors
    if palette is None:
        palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return list(islice(cycle(palette), len(levels)))


class _TableFacetGrid:
    """Create Matplotlib panels and dispatch callbacks for tabular facets."""

    def __init__(
        self,
        data,
        *,
        row,
        col,
        hue,
        col_wrap,
        height,
        aspect,
        subplot_kws,
        row_order=None,
        col_order=None,
        hue_order=None,
        sharex=True,
        sharey=True,
        palette=None,
        hue_kws=None,
        legend_out=True,
        despine=True,
        margin_titles=False,
        dropna=False,
        xlim=None,
        ylim=None,
        gridspec_kws=None,
    ):
        if col_wrap is not None and (row is not None or col is None or col_wrap < 1):
            raise ValueError("col_wrap requires column facets and a positive width.")
        self.data = data
        self._row, self._col, self._hue = row, col, hue
        self._wrap = col_wrap
        self._dropna = dropna
        self._legend_out = legend_out
        self._margin_titles = margin_titles
        self.hue_kws = hue_kws or {}
        self.row_names = _facet_levels(data, row, row_order)
        self.col_names = _facet_levels(data, col, col_order)
        self.hue_names = (
            _facet_levels(data, hue, hue_order) if hue is not None else None
        )

        self._colors = _facet_colors(palette, self.hue_names or [None])

        nrow, ncol = len(self.row_names) or 1, len(self.col_names) or 1
        self._facets = list(product(range(nrow), range(ncol)))
        if col_wrap:
            nrow, ncol = ceil(ncol / col_wrap), col_wrap
        self.fig, self.axs = plt.subplots(
            nrow,
            ncol,
            squeeze=False,
            sharex=sharex,
            sharey=sharey,
            figsize=(ncol * height * aspect, nrow * height),
            subplot_kw=subplot_kws,
            gridspec_kw=gridspec_kws,
        )
        self.figure = self.fig
        self.axes = self.axs.ravel()[: len(self._facets)] if col_wrap else self.axs
        self.axes_dict = {}
        for (i, j), ax in zip(self._facets, self.axs.flat):
            key = tuple(
                names[k]
                for names, k in ((self.row_names, i), (self.col_names, j))
                if names
            )
            if key:
                self.axes_dict[key[0] if len(key) == 1 else key] = ax
            if despine:
                ax.spines[["right", "top"]].set_visible(False)
            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)
        for ax in self.axs.ravel()[len(self._facets) :]:
            ax.set_visible(False)
        if col_wrap and sharex:
            for ax in self._outer_axes("x"):
                ax.tick_params(labelbottom=True)
        self._margin_annotations = []
        self._legend = None
        self.set_titles()

    @property
    def ax(self):
        if self.axes.size != 1:
            raise AttributeError("Use .axes for a grid with multiple facets.")
        return self.axes.flat[0]

    def facet_axis(self, row_i, col_j, modify_state=True):
        ax = self.axes[col_j] if self._wrap else self.axes[row_i, col_j]
        if modify_state:
            plt.sca(ax)
        return ax

    def facet_data(self):
        """Yield each row, column and hue subset with its facet indices."""

        def masks(column, levels):
            if column is None:
                return [np.ones(len(self.data), dtype=bool)]
            return [self.data[column].eq(level).fillna(False) for level in levels]

        groups = zip(
            (self._row, self._col, self._hue),
            (self.row_names, self.col_names, self.hue_names),
        )
        for (i, row), (j, col), (k, hue) in product(
            *(enumerate(masks(column, levels)) for column, levels in groups)
        ):
            yield (i, j, k), self.data.loc[row & col & hue]

    def _map(self, func, args, kwargs, *, frame):
        color = kwargs.pop("color", None)
        for (i, j, k), subset in self.facet_data():
            if subset.empty:
                continue
            self.facet_axis(i, j)
            options = dict(kwargs, color=self._colors[k] if color is None else color)
            options.update({name: values[k] for name, values in self.hue_kws.items()})
            if self.hue_names is not None:
                options["label"] = str(self.hue_names[k])
            if frame:
                options["data"] = subset.dropna() if self._dropna else subset
                func(*args, **options)
            else:
                columns = subset[list(args)]
                if self._dropna:
                    columns = columns.dropna()
                values = (column for _, column in columns.items())
                if getattr(func, "__module__", "").startswith("matplotlib"):
                    values = (column.to_numpy() for _, column in columns.items())
                func(*values, **options)
        labels = [kwargs.get("x"), kwargs.get("y")]
        labels[: len(args[:2])] = args[:2]
        self.set_axis_labels(*labels)
        return self.set_titles().tight_layout()

    def map(self, func, *args, **kwargs):
        """Call a plotting function with the selected columns as arrays."""
        return self._map(func, args, kwargs, frame=False)

    def map_dataframe(self, func, *args, **kwargs):
        """Call a plotting function with a subset passed as the data keyword."""
        return self._map(func, args, kwargs, frame=True)

    def _outer_axes(self, axis):
        if axis == "y":
            return [ax for ax in self.axs[:, 0] if ax.get_visible()]
        bottom = []
        for column in self.axs.T:
            visible = [ax for ax in column if ax.get_visible()]
            if visible:
                bottom.append(visible[-1])
        return bottom

    def set_xlabels(self, label=None, clear_inner=True, **kwargs):
        return self._set_labels("x", label, clear_inner, **kwargs)

    def set_ylabels(self, label=None, clear_inner=True, **kwargs):
        return self._set_labels("y", label, clear_inner, **kwargs)

    def _set_labels(self, axis, label, clear_inner, **kwargs):
        if label is not None:
            outer = self._outer_axes(axis)
            for ax in self.axes.flat:
                if ax in outer or clear_inner:
                    getattr(ax, f"set_{axis}label")(
                        label if ax in outer else "", **kwargs
                    )
        return self

    def set_axis_labels(self, x_var=None, y_var=None, clear_inner=True, **kwargs):
        return self.set_xlabels(x_var, clear_inner, **kwargs).set_ylabels(
            y_var, clear_inner, **kwargs
        )

    def set_titles(self, template=None, row_template=None, col_template=None, **kwargs):
        row_template = row_template or "{row_var} = {row_name}"
        col_template = col_template or "{col_var} = {col_name}"
        for artist in self._margin_annotations:
            artist.remove()
        self._margin_annotations.clear()
        for i, j in self._facets:
            ax = self.facet_axis(i, j, modify_state=False)
            values = {
                "row_var": self._row,
                "col_var": self._col,
                "row_name": self.row_names[i] if self.row_names else None,
                "col_name": self.col_names[j] if self.col_names else None,
            }
            if self._margin_titles:
                if i == 0 and self.col_names:
                    ax.set_title(col_template.format(**values), **kwargs)
                if j == self.axs.shape[1] - 1 and self.row_names:
                    self._margin_annotations.append(
                        ax.annotate(
                            row_template.format(**values),
                            xy=(1.02, 0.5),
                            xycoords="axes fraction",
                            rotation=270,
                            va="center",
                            **kwargs,
                        )
                    )
            else:
                parts = [
                    part
                    for names, part in (
                        (self.row_names, row_template),
                        (self.col_names, col_template),
                    )
                    if names
                ]
                if parts:
                    ax.set_title(
                        (template or " | ".join(parts)).format(**values), **kwargs
                    )
        return self

    def add_legend(self, legend_data=None, title=None, label_order=None, **kwargs):
        """Add a legend using handles from all the facet axes."""
        if legend_data is None:
            legend_data = {}
            for ax in self.axes.flat:
                handles, labels = ax.get_legend_handles_labels()
                legend_data.update(zip(labels, handles))
        if label_order is None:
            label_order = self.hue_names if self.hue_names is not None else legend_data
        labels = list(map(str, label_order))
        handles = [legend_data.get(label, Patch(visible=False)) for label in labels]
        title = self._hue if title is None else title
        if self._legend is not None:
            self._legend.remove()
        if self._legend_out:
            kwargs.setdefault("loc", "center right")
            self._legend = self.fig.legend(handles, labels, title=title, **kwargs)
            width = self._legend.get_window_extent().width / self.fig.bbox.width
            self._layout_rect = (0, 0, max(0.1, 1 - width), 1)
            self.tight_layout()
        else:
            self._legend = self.axes.flat[0].legend(
                handles, labels, title=title, **kwargs
            )
        return self

    def tight_layout(self, *args, **kwargs):
        kwargs.setdefault("rect", getattr(self, "_layout_rect", None))
        self.fig.tight_layout(*args, **kwargs)
        return self

    def set(self, **kwargs):
        for ax in self.axes.flat:
            ax.set(**kwargs)
        return self

    def savefig(self, *args, **kwargs):
        kwargs.setdefault("bbox_inches", "tight")
        return self.fig.savefig(*args, **kwargs)


class FacetGridPlot(BasePlot):
    """Creates a facet grid plot.

    Tabular facets use Matplotlib axes and apply a callback to each group.
    """

    def __init__(
        self,
        data: Any,
        row: str | None = None,
        col: str | None = None,
        hue: str | None = None,
        col_wrap: int | None = None,
        size: float | None = None,
        aspect: float = 1,
        subplot_kws: dict[str, Any] | None = None,
        style: str | None = "wiley",
        **kwargs: Any,
    ) -> None:
        """Initializes the facet grid.

        Parameters
        ----------
        data : Any
            The data to plot.
        row : str, optional
            Variable to map to row facets, by default None.
        col : str, optional
            Variable to map to column facets, by default None.
        hue : str, optional
            Variable to map to color mapping, by default None.
        col_wrap : int, optional
            Number of columns before wrapping, by default None.
        size : float, optional
            Height (in inches) of each facet, by default 3.
            Aligns with Xarray convention.
        aspect : float, optional
            Aspect ratio of each facet, by default 1.
        subplot_kws : dict, optional
            Keyword arguments for subplots (e.g. projection).
        style : str, optional
            Style name to apply, by default "wiley".
        **kwargs : Any
            Additional options for tabular facets, including facet orders, palette,
            shared axes, and legend placement.
            `height` is supported as an alias for `size`.
        """
        # Apply style
        if style:
            set_style(style)

        # Accept both conventions for facet height.
        height = kwargs.pop("height", 3)
        self.size = height if size is None else size
        self.aspect = aspect

        # Store facet parameters
        self.row = row
        self.col = col
        self.hue = hue
        self.col_wrap = col_wrap

        # Aero Protocol: Preserve lazy Xarray objects
        self.raw_data = data
        self.is_xarray = isinstance(data, (xr.DataArray, xr.Dataset))

        if self.is_xarray:
            self.data = data  # Keep as Xarray
            # For Xarray, we can use xarray.plot.FacetGrid
            # We delay creation to SpatialFacetGridPlot if possible,
            # or initialize it here if we have enough info.
            self.grid = None
            if col or row:
                try:
                    from xarray.plot.facetgrid import FacetGrid as xrFacetGrid

                    self.grid = xrFacetGrid(
                        data,
                        col=col,
                        row=row,
                        col_wrap=col_wrap,
                        subplot_kws=subplot_kws,
                    )
                    # Initialize default titles for Xarray
                    self.grid.set_titles()
                except (ImportError, TypeError, AttributeError):
                    pass
        else:
            self.data = to_dataframe(data).reset_index()
            self.grid = _TableFacetGrid(
                self.data,
                row=self.row,
                col=self.col,
                hue=self.hue,
                col_wrap=self.col_wrap,
                height=self.size,
                aspect=self.aspect,
                subplot_kws=subplot_kws,
                **kwargs,
            )

        # Unified BasePlot initialization
        axes = _get_grid_axes(self.grid)
        if axes is not None:
            super().__init__(fig=self.grid.fig, ax=axes.flatten()[0], style=style)
        else:
            super().__init__(style=style)

        # For compatibility with tests, also store as 'g'
        self.g = self.grid

    def map_dataframe(self, plot_func: Callable, *args: Any, **kwargs: Any) -> None:
        """Maps a plotting function to the facet grid.

        Args:
            plot_func (function): The plotting function to map.
            *args: Positional arguments to pass to the plotting function.
            **kwargs: Keyword arguments to pass to the plotting function.
        """
        self.grid.map_dataframe(plot_func, *args, **kwargs)

    def set_titles(self, *args, **kwargs):
        """Sets the titles of the facet grid.

        Args:
            *args: Positional arguments to pass to `set_titles`.
            **kwargs: Keyword arguments to pass to `set_titles`.
        """
        self.grid.set_titles(*args, **kwargs)

    def save(self, filename, **kwargs):
        """Saves the plot to a file.

        Args:
            filename (str): The name of the file to save the plot to.
            **kwargs: Additional keyword arguments to pass to `savefig`.
        """
        self.fig.savefig(filename, **kwargs)

    def plot(self, plot_func=None, *args, **kwargs):
        """Plots the data using the FacetGrid.

        Args:
            plot_func (function, optional): The plotting function to use.
            *args: Positional arguments to pass to the plotting function.
            **kwargs: Keyword arguments to pass to the plotting function.
        """
        if plot_func is not None:
            self.grid.map(plot_func, *args, **kwargs)

    def close(self):
        """Closes the plot."""
        plt.close(self.fig)


class SpatialFacetGridPlot(FacetGridPlot):
    """Geospatial version of FacetGridPlot."""

    def __init__(
        self,
        data: Any,
        *,
        row: str | None = None,
        col: str | None = None,
        col_wrap: int | None = None,
        projection: ccrs.Projection | None = None,
        size: float | None = None,
        aspect: float = 1.2,
        style: str | None = "wiley",
        **kwargs: Any,
    ) -> None:
        """Initialize Spatial Facet Grid.

        Parameters
        ----------
        data : Any
            Geospatial data to plot. Preferred format is xr.DataArray or xr.Dataset.
        row : str, optional
            Dimension/variable to map to rows.
        col : str, optional
            Dimension/variable to map to columns.
        col_wrap : int, optional
            Wrap columns at this number.
        projection : ccrs.Projection, optional
            Cartopy projection for the maps. Defaults to PlateCarree.
        size : float, optional
            Height (in inches) of each facet, by default 4.
            Aligns with Xarray convention.
        aspect : float, optional
            Aspect ratio of each facet, by default 1.2.
        style : str, optional
            Style name to apply, by default "wiley".
        **kwargs : Any
            Additional arguments for FacetGrid.
        """
        self.original_data = data
        import cartopy.crs as ccrs

        self.projection = projection or ccrs.PlateCarree()

        # Handle xr.Dataset by converting to DataArray if faceting by variable
        # or if there is only one data variable.
        self.is_dataset = isinstance(data, xr.Dataset)
        if self.is_dataset:
            if row == "variable" or col == "variable":
                data = data.to_array(dim="variable", name="value")
            elif len(data.data_vars) == 1:
                # Auto-select the only variable to ensure map_dataarray works
                data = data[list(data.data_vars)[0]]

        # Aligns with Xarray's default size for maps
        size = size or kwargs.pop("height", 4)

        # Call FacetGridPlot init which handles the two-track branching
        super().__init__(
            data,
            row=row,
            col=col,
            col_wrap=col_wrap,
            size=size,
            aspect=aspect,
            subplot_kws={"projection": self.projection},
            style=style,
            **kwargs,
        )

        # Set default titles if grid is already created (Pandas track)
        if self.grid:
            self._set_default_titles()

    def _set_default_titles(self) -> None:
        """Format facet titles with metadata and date-time."""
        axes = _get_grid_axes(self.grid)
        if axes is None:
            return

        for ax in axes.flatten():
            if ax is None or not ax.get_visible():
                continue
            title = ax.get_title()

            # Handle titles that might have multiple facets (e.g. "row = val | col = val")
            parts = title.split(" | ")
            new_parts = []

            for part in parts:
                if " = " in part:
                    # Use split(" = ", 1) to handle values that might contain " = "
                    dim_val = part.split(" = ", 1)
                    if len(dim_val) == 2:
                        dim, val = dim_val
                        dim = dim.strip()
                        val = val.strip()

                        # Handle date-time formatting
                        try:
                            dt = pd.to_datetime(val)
                            val = dt.strftime("%Y-%m-%d %H:%M")
                        except (ValueError, TypeError):
                            pass

                        # Handle long_name for dimensions/variables
                        if dim == "variable" and self.is_dataset:
                            try:
                                # self.original_data is the original Dataset
                                var_obj = self.original_data[val]
                                long_name = var_obj.attrs.get("long_name", val)
                                units = var_obj.attrs.get("units", "")
                                val = f"{long_name} ({units})" if units else long_name
                                dim = ""
                            except (KeyError, AttributeError):
                                pass
                        elif dim in getattr(self.original_data, "coords", ()):
                            try:
                                coord_obj = self.original_data.coords[dim]
                                long_name = coord_obj.attrs.get("long_name", dim)
                                dim = long_name
                            except (KeyError, AttributeError):
                                pass

                        new_parts.append(f"{dim} {val}".strip())
                    else:
                        new_parts.append(part)
                else:
                    new_parts.append(part)

            ax.set_title(" | ".join(new_parts))

    def add_map_features(self, **kwargs: Any) -> None:
        """Add cartopy features to all facets.

        Parameters
        ----------
        **kwargs : Any
            Keyword arguments passed to SpatialPlot.add_features.
            Default is coastlines=True.
        """
        from .spatial import SpatialPlot

        if "coastlines" not in kwargs:
            kwargs["coastlines"] = True

        axes = _get_grid_axes(self.grid)
        if axes is None:
            return

        for ax in axes.flatten():
            if ax is None or not ax.get_visible():
                continue
            # Use SpatialPlot's feature logic on each axis
            SpatialPlot(ax=ax, projection=self.projection, **kwargs)

    def map_monet(
        self,
        plotter_class: type,
        *,
        x: str | None = None,
        y: str | None = None,
        var_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Map a monet-plots spatial plotter to the grid.

        Parameters
        ----------
        plotter_class : type
            A class from monet_plots.plots (e.g., SpatialImshowPlot).
        x : str, optional
            Column name for longitude.
        y : str, optional
            Column name for latitude.
        var_name : str, optional
            The variable name to plot.
        **kwargs : Any
            Arguments passed to the plotter and map features.
        """
        # Separate feature kwargs
        feature_keys = [
            "coastlines",
            "states",
            "countries",
            "ocean",
            "land",
            "lakes",
            "rivers",
            "borders",
            "gridlines",
            "extent",
            "resolution",
        ]
        feature_kwargs = {k: kwargs.pop(k) for k in feature_keys if k in kwargs}
        add_shared_cb = kwargs.pop("add_colorbar", False)

        if self.is_xarray:
            # Track A: Xarray-native faceting (Lazy by Default)
            from .spatial import SpatialPlot

            # Identify coordinates if not provided
            if x is None or y is None:
                x_id, y_id = SpatialPlot._identify_coords(self.data)
                x = x or x_id
                y = y or y_id

            # Use Xarray's native plotting which handles faceting
            plot_type = "imshow"
            if "Contour" in plotter_class.__name__:
                plot_type = "contourf"

            # Prepare plotting arguments
            plot_kwargs = kwargs.copy()
            plot_kwargs.setdefault("transform", ccrs.PlateCarree())
            if add_shared_cb:
                plot_kwargs.setdefault("add_colorbar", True)
                plot_kwargs.setdefault("cbar_kwargs", {"orientation": "horizontal"})

            # Select data variable if it's a Dataset
            plot_data = self.data
            if isinstance(plot_data, xr.Dataset):
                if var_name:
                    plot_data = plot_data[var_name]
                else:
                    # Pick the first data variable that is not a facet dimension
                    for v in plot_data.data_vars:
                        if v not in [self.col, self.row]:
                            plot_data = plot_data[v]
                            break

            # Trigger Xarray's facet grid
            # If we already have a grid, we can use it
            if self.grid is not None:
                import xarray.plot as xplt

                func = xplt.imshow if plot_type == "imshow" else xplt.contourf

                # Check if the grid's data is a Dataset.
                # If so, map_dataarray might fail in some xarray versions.
                # We use map with a selection wrapper instead.
                grid_data = getattr(self.grid, "data", None)
                if isinstance(grid_data, xr.Dataset):
                    # var_name was selected above from self.data or passed in
                    def _mapped_xr_plot(ds_subset, x_coord, y_coord, **inner_kwargs):
                        func(ds_subset[var_name], x=x_coord, y=y_coord, **inner_kwargs)

                    self.grid.map(_mapped_xr_plot, x, y, **plot_kwargs)
                elif hasattr(self.grid, "map_dataarray"):
                    if (
                        isinstance(grid_data, xr.DataArray)
                        and grid_data.chunks is not None
                    ):
                        # Xarray needs all panel values for its shared scale. Use
                        # one snapshot for the scale and individual panel renders.
                        self.grid.data = grid_data.compute()
                        try:
                            self.grid.map_dataarray(func, x, y, **plot_kwargs)
                        finally:
                            self.grid.data = grid_data
                    else:
                        self.grid.map_dataarray(func, x, y, **plot_kwargs)
                else:
                    self.grid.map(func, x, y, **plot_kwargs)
            else:
                xr_plot_func = getattr(plot_data.plot, plot_type)
                self.grid = xr_plot_func(
                    x=x,
                    y=y,
                    col=self.col,
                    row=self.row,
                    col_wrap=self.col_wrap,
                    subplot_kws={"projection": self.projection},
                    **plot_kwargs,
                )

            # Update BasePlot attributes
            self.fig = self.grid.fig
            axes = _get_grid_axes(self.grid)
            self.ax = axes.flatten()[0]
            self.g = self.grid

            # Add features to all facets
            self.add_map_features(**feature_kwargs)
            self._set_default_titles()

        else:
            # Track B: Tabular faceting
            x = x or "lon"
            y = y or "lat"
            if var_name is None:
                if "variable" in self.data.columns:
                    var_name = "value"
                elif isinstance(self.raw_data, xr.DataArray):
                    var_name = self.raw_data.name
                elif isinstance(self.raw_data, xr.Dataset):
                    var_name = list(self.raw_data.data_vars)[0]

            def _mapped_plot(*args, **kwargs_inner):
                data_df = kwargs_inner.pop("data")
                ax = plt.gca()
                temp_da = data_df.set_index([y, x]).to_xarray()[var_name]
                plotter = plotter_class(temp_da, ax=ax, **kwargs_inner)
                plotter.plot()

            self.map_dataframe(_mapped_plot, **kwargs)
            self.add_map_features(**feature_kwargs)
            if add_shared_cb:
                self._add_shared_colorbar(**kwargs)

    def _add_shared_colorbar(self, **kwargs: Any) -> None:
        """Add a shared colorbar to the figure."""
        # Find the last mappable object in the facets and the last valid axis
        mappable = None
        target_ax = None
        axes = _get_grid_axes(self.grid)
        if axes is None:
            return

        for ax in reversed(axes.flatten()):
            if ax is None or not ax.get_visible():
                continue
            if target_ax is None:
                target_ax = ax
            if ax.collections and mappable is None:
                mappable = ax.collections[0]
            if ax.images and mappable is None:
                mappable = ax.images[0]

        if mappable and target_ax:
            # Add colorbar to the last valid axis
            self.add_colorbar(
                mappable,
                ax=target_ax,
                label=kwargs.get("label", ""),
            )
