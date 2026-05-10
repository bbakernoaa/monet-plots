# src/monet_plots/plots/facet_grid.py
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xarray as xr

from ..plot_utils import to_dataframe
from ..style import set_style
from .base import BasePlot

if TYPE_CHECKING:
    import cartopy.crs as ccrs


class FacetGridPlot(BasePlot):
    """Creates a facet grid plot.

    This class creates a facet grid plot using seaborn's FacetGrid.
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
            Additional keyword arguments to pass to `FacetGrid`.
            `height` is supported as an alias for `size` (Seaborn convention).
        """
        # Apply style
        if style:
            set_style(style)

        # Handle size/height alignment (Xarray vs Seaborn)
        self.size = size or kwargs.pop("height", 3)
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
            # Create the Seaborn FacetGrid
            self.grid = sns.FacetGrid(
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
        axes = getattr(self.grid, "axs", None)
        if axes is None:
            axes = getattr(self.grid, "axes", None)
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
        axes = getattr(self.grid, "axs", None)
        if axes is None:
            axes = getattr(self.grid, "axes", None)
        if axes is None:
            return

        for ax in axes.flatten():
            if ax is None:
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

                        # Handle long_name and units for dimensions/variables
                        target_obj = None
                        if dim == "variable" and self.is_dataset:
                            try:
                                target_obj = self.original_data[val]
                            except (KeyError, AttributeError):
                                pass
                        elif dim in self.original_data.coords:
                            target_obj = self.original_data.coords[dim]
                        elif (
                            hasattr(self.original_data, "data_vars")
                            and dim in self.original_data.data_vars
                        ):
                            target_obj = self.original_data.data_vars[dim]

                        if target_obj is not None:
                            long_name = target_obj.attrs.get(
                                "long_name", dim if dim != "variable" else val
                            )
                            units = target_obj.attrs.get("units", "")

                            if dim == "variable":
                                val = f"{long_name} ({units})" if units else long_name
                                dim = ""
                            else:
                                dim = long_name
                                if units and not any(
                                    u in val for u in [f"({units})", f"[{units}]"]
                                ):
                                    val = f"{val} ({units})"

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

        axes = getattr(self.grid, "axs", None)
        if axes is None:
            axes = getattr(self.grid, "axes", None)
        if axes is None:
            return

        for ax in axes.flatten():
            if ax is None:
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
            # Identify coordinates if not provided
            if x is None or y is None:
                from .spatial import SpatialPlot

                sp = SpatialPlot(style=None)
                x_id, y_id = sp._identify_coords(self.data)
                x = x or x_id
                y = y or y_id

            # Prepare plotting arguments
            plot_kwargs = kwargs.copy()
            # If we want a shared colorbar, we don't want each plotter to add its own
            plot_kwargs["add_colorbar"] = False

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

            # Create the FacetGrid if it doesn't exist
            if self.grid is None:
                from xarray.plot.facetgrid import FacetGrid as xrFacetGrid

                self.grid = xrFacetGrid(
                    plot_data,
                    col=self.col,
                    row=self.row,
                    col_wrap=self.col_wrap,
                    subplot_kws={"projection": self.projection},
                )

            # Identify variables needed by the plotter
            vars_to_map = []
            if var_name:
                vars_to_map = [var_name]
            elif "col1" in kwargs and "col2" in kwargs:
                # For SpatialBiasScatterPlot
                vars_to_map = [kwargs["col1"], kwargs["col2"]]
            elif isinstance(plot_data, xr.DataArray) and plot_data.name:
                vars_to_map = [plot_data.name]
            elif isinstance(plot_data, xr.Dataset):
                # Use all data variables if not specified
                vars_to_map = list(plot_data.data_vars)

            # Use a wrapper to instantiate the plotter class for each facet
            def _monet_wrapper(*args, **inner_kwargs):
                ax = inner_kwargs.pop("ax", plt.gca())
                # Remove xarray's extra internal kwargs, but preserve levels and norm
                # if they were explicitly provided in the original kwargs.
                for k in ["x", "y", "add_labels", "_is_facetgrid", "extend"]:
                    inner_kwargs.pop(k, None)

                # If levels/norm were not in kwargs but are in inner_kwargs (from xarray's auto-scale),
                # we only keep them if they are not None.
                if "levels" in inner_kwargs and "levels" not in kwargs:
                    if inner_kwargs["levels"] is None:
                        inner_kwargs.pop("levels")
                if "norm" in inner_kwargs and "norm" not in kwargs:
                    if inner_kwargs["norm"] is None:
                        inner_kwargs.pop("norm")

                if len(args) == 1:
                    data_subset = args[0]
                elif len(args) > 1:
                    # Check if args are numpy arrays (sometimes happens in xarray versions)
                    processed_args = []
                    for i, arg in enumerate(args):
                        if isinstance(arg, np.ndarray):
                            var_name_at_index = vars_to_map[i]
                            # Try to find matching DataArray in the original data to copy coordinates
                            # self.data is the full data (DataArray or Dataset)
                            coords = None
                            dims = None
                            if isinstance(self.data, xr.Dataset):
                                if var_name_at_index in self.data:
                                    # Subset coordinates to only those that match the arg shape
                                    full_da = self.data[var_name_at_index]
                                    dims = [
                                        d
                                        for d in full_da.dims
                                        if d not in [self.col, self.row]
                                    ]
                                    coords = {
                                        c: full_da.coords[c]
                                        for c in full_da.coords
                                        if all(
                                            d in dims for d in full_da.coords[c].dims
                                        )
                                    }
                            elif isinstance(self.data, xr.DataArray):
                                dims = [
                                    d
                                    for d in self.data.dims
                                    if d not in [self.col, self.row]
                                ]
                                coords = {
                                    c: self.data.coords[c]
                                    for c in self.data.coords
                                    if all(d in dims for d in self.data.coords[c].dims)
                                }

                            da = xr.DataArray(
                                arg, name=var_name_at_index, coords=coords, dims=dims
                            )
                            processed_args.append(da)
                        else:
                            processed_args.append(arg)

                    # Merge multiple DataArrays back into a Dataset
                    data_subset = xr.merge(processed_args, compat="override")
                else:
                    # Fallback if no args passed
                    data_subset = None

                # Create the plotter instance
                plotter = plotter_class(
                    data_subset, ax=ax, _is_facetgrid=True, **inner_kwargs
                )
                # Call its plot method
                plotter.plot()

            # Map the wrapper to the grid
            if isinstance(self.grid.data, xr.Dataset):
                self.grid.map(_monet_wrapper, *vars_to_map, **plot_kwargs)
            elif hasattr(self.grid, "map_dataarray"):
                self.grid.map_dataarray(_monet_wrapper, x, y, **plot_kwargs)
            else:
                self.grid.map(_monet_wrapper, **plot_kwargs)

            # Update BasePlot attributes
            self.fig = self.grid.fig
            axes = getattr(self.grid, "axs", None)
            if axes is None:
                axes = getattr(self.grid, "axes", None)
            self.ax = axes.flatten()[0]
            self.g = self.grid

            # Add features to all facets
            self.add_map_features(**feature_kwargs)
            self._set_default_titles()

        else:
            # Track B: Seaborn-based faceting (Eager/Pandas)
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
        axes = getattr(self.grid, "axs", None)
        if axes is None:
            axes = getattr(self.grid, "axes", None)
        if axes is None:
            return

        for ax in reversed(axes.flatten()):
            if ax is None:
                continue
            if target_ax is None:
                target_ax = ax
            if ax.collections and mappable is None:
                mappable = ax.collections[0]
            if ax.images and mappable is None:
                mappable = ax.images[0]

        if mappable and target_ax:
            # For contourf, we might want to use the levels if available
            levels = kwargs.get("levels")
            if (
                levels is not None
                and not isinstance(levels, int)
                and hasattr(levels, "__len__")
            ):
                # Use discrete colorbar if levels are provided
                from ..colorbars import colorbar_index

                colorbar_index(
                    len(levels) - 1,
                    mappable.get_cmap(),
                    minval=levels[0],
                    maxval=levels[-1],
                    ax=target_ax,
                    label=kwargs.get("label", ""),
                )
            else:
                # Add colorbar to the last valid axis
                self.add_colorbar(
                    mappable,
                    ax=target_ax,
                    label=kwargs.get("label", ""),
                )
