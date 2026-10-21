"""Faceting for tabular plots and geographically comparable spatial panels."""

from __future__ import annotations

from inspect import Parameter, signature
from typing import Any, Callable

import cartopy.crs as ccrs
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib.colors import BoundaryNorm, LogNorm, Normalize
from xarray.plot import FacetGrid as XarrayFacetGrid

from ..geo import identify_coords, validate_data_crs, variable_label
from ..plot_utils import normalize_data, to_dataframe
from ..style import style_context
from .base import BasePlot


class FacetGridPlot(BasePlot):
    """A consistent figure wrapper around seaborn and xarray FacetGrid."""

    def __init__(
        self,
        data: Any,
        row: str | None = None,
        col: str | None = None,
        hue: str | None = None,
        col_wrap: int | None = None,
        size: float | None = None,
        aspect: float = 1,
        subplot_kws: dict | None = None,
        style: str | None = "wiley",
        **kwargs,
    ):
        self.row, self.col, self.hue, self.col_wrap = row, col, hue, col_wrap
        self.size = size if size is not None else kwargs.pop("height", 3)
        self.aspect = aspect
        self.raw_data = data
        self.is_xarray = isinstance(data, (xr.DataArray, xr.Dataset))
        if row is not None and col_wrap is not None:
            raise ValueError("col_wrap cannot be combined with row faceting.")
        if self.is_xarray:
            if row is None and col is None:
                raise ValueError("Specify row or col to create an xarray facet grid.")
            if hue is not None:
                raise ValueError("hue is supported for tabular facets only.")
            self.data = normalize_data(data)
            # Fail with the actual xarray error; never silently create an empty figure.
            self.grid = XarrayFacetGrid(
                self.data,
                row=row,
                col=col,
                col_wrap=col_wrap,
                size=self.size,
                aspect=aspect,
                subplot_kws=subplot_kws,
                **kwargs,
            )
            self.grid.set_titles()
        else:
            self.data = to_dataframe(data).reset_index()
            self.grid = sns.FacetGrid(
                self.data,
                row=row,
                col=col,
                hue=hue,
                col_wrap=col_wrap,
                height=self.size,
                aspect=aspect,
                subplot_kws=subplot_kws,
                **kwargs,
            )
        axes = self.grid.axs if self.is_xarray else self.grid.axes
        super().__init__(fig=self.grid.fig, ax=axes.flat[0], style=style)
        self.axes = axes
        self.g = self.grid

    def map_dataframe(self, plot_func: Callable, *args, **kwargs):
        """Map a seaborn-style function over tabular subsets."""
        if self.is_xarray:
            raise TypeError(
                "map_dataframe requires tabular data; use map_monet for spatial xarray facets."
            )
        self.grid.map_dataframe(plot_func, *args, **kwargs)
        return self

    def set_titles(self, *args, **kwargs):
        """Set panel titles with the underlying backend's formatting options."""
        self.grid.set_titles(*args, **kwargs)
        return self

    def plot(self, plot_func=None, *args, **kwargs):
        """Map a function over the facet grid."""
        if plot_func is not None:
            self.grid.map(plot_func, *args, **kwargs)
        return self


class SpatialFacetGridPlot(FacetGridPlot):
    """Spatial panels with explicit CRS and scientifically comparable scales.

    ``colorbar='shared'`` (default) uses one norm across all panels. ``'each'``
    permits independent scales, including variables with incompatible units.
    ``False`` hides colorbars while retaining a common scale. Initialization
    preserves lazy arrays; rendering computes the global scale and panel values.
    """

    def __init__(
        self,
        data,
        *,
        row=None,
        col=None,
        col_wrap=None,
        projection=None,
        data_crs=None,
        extent_crs=None,
        size=None,
        aspect=1.2,
        style="wiley",
        colorbar="shared",
        _plotter_class=None,
        _plotter_kwargs=None,
        **kwargs,
    ):
        self.original_data = data
        self.is_dataset = isinstance(data, xr.Dataset)
        self.projection = projection or ccrs.PlateCarree()
        self._data_crs_explicit = data_crs is not None
        self.data_crs = data_crs or ccrs.PlateCarree()
        self.extent_crs = extent_crs or ccrs.PlateCarree()
        self.default_plotter = _plotter_class
        self.plotter_kwargs = _plotter_kwargs or {}
        self.colorbar_mode = colorbar
        self.plotters, self.mappables = [], []
        self._variable_facet = self.is_dataset and "variable" in (row, col)
        if self._variable_facet:
            if len({frozenset(v.dims) for v in data.data_vars.values()}) > 1:
                raise ValueError(
                    "Variables must have matching dimensions to facet; select compatible fields or regrid explicitly."
                )
            # Keep metadata on original variables for labels and unit validation.
            data = data.to_array(dim="variable", name="value")
        elif self.is_dataset and len(data.data_vars) == 1:
            data = data[next(iter(data.data_vars))]
        height = kwargs.pop("height", 4)
        size = size if size is not None else height
        grid_keys = (
            {"figsize", "sharex", "sharey"}
            if isinstance(data, (xr.Dataset, xr.DataArray))
            else {
                "row_order",
                "col_order",
                "hue",
                "hue_order",
                "palette",
                "sharex",
                "sharey",
                "margin_titles",
                "despine",
                "dropna",
                "legend_out",
                "gridspec_kws",
            }
        )
        grid_options = {
            key: kwargs.pop(key) for key in list(kwargs) if key in grid_keys
        }
        self.render_options = {**(self.plotter_kwargs.get("plotargs") or {}), **kwargs}
        super().__init__(
            data,
            row=row,
            col=col,
            col_wrap=col_wrap,
            size=size,
            aspect=aspect,
            subplot_kws={"projection": self.projection},
            style=style,
            **grid_options,
        )
        self.fig.set_layout_engine("constrained")
        self._set_default_titles()

    def _panels(self):
        if self.is_xarray:
            for ax, selection in zip(self.axes.flat, self.grid.name_dicts.flat):
                if selection is None:
                    ax.set_visible(False)
                    continue
                subset = self.data.sel(selection)
                if self._variable_facet and "variable" in selection:
                    subset = subset.assign_attrs(
                        self.original_data[selection["variable"]].attrs
                    )
                    subset.name = str(selection["variable"])
                yield ax, subset, selection
        else:
            for (i, j, _), subset in self.grid.facet_data():
                if subset.empty:
                    continue
                ax = self.axes.flat[j] if self.col_wrap else self.axes[i, j]
                selection = {}
                if self.row:
                    selection[self.row] = self.grid.row_names[i]
                if self.col:
                    selection[self.col] = self.grid.col_names[j]
                yield ax, subset, selection

    def _set_default_titles(self):
        if not self.is_xarray:
            return
        for annotation in self.grid.row_labels:
            if annotation is not None and annotation.axes is not None:
                annotation.remove()
        self.grid.row_labels = []
        for ax, _, selection in self._panels():
            parts = []
            for dim, value in selection.items():
                if dim == "variable" and self._variable_facet:
                    parts.append(variable_label(self.original_data[value]))
                else:
                    coord = self.original_data.coords.get(dim)
                    label = (
                        coord.attrs.get("long_name", dim) if coord is not None else dim
                    )
                    if isinstance(value, (np.datetime64, pd.Timestamp)):
                        value = pd.Timestamp(value).strftime("%Y-%m-%d %H:%M")
                    parts.append(f"{label} = {value}")
            ax.set_title(" | ".join(parts))

    def add_map_features(self, **kwargs):
        """Add features to occupied panels, leaving wrapped empty cells hidden."""
        from .spatial import SpatialPlot

        for ax, _, _ in self._panels():
            SpatialPlot(
                ax=ax,
                data_crs=self.data_crs,
                extent_crs=self.extent_crs,
                style=self._style,
                **kwargs,
            )
        return self

    def plot(self, plotter_class=None, **kwargs):
        """Render using the originating spatial class, or SpatialMeshPlot."""
        from .spatial_mesh import SpatialMeshPlot

        return self.map_monet(
            plotter_class or self.default_plotter or SpatialMeshPlot, **kwargs
        )

    def map_monet(
        self, plotter_class: type, *, x=None, y=None, var_name=None, **kwargs
    ):
        """Render each subset using a spatial plotter, returning this grid.

        Constructor-only arguments (e.g. col1/col2, gridobj) are routed to the
        constructor. Rendering options (e.g. norm, cmap) are sent to ``plot``.
        ``colorbar`` accepts 'shared', 'each', or False; ``add_colorbar`` is a
        boolean compatibility alias. ``cbar_kwargs`` configure the real mappable.
        """
        options = {**self.render_options, **kwargs}
        mode = options.pop("colorbar", self.colorbar_mode)
        if "add_colorbar" in options:
            mode = "shared" if options.pop("add_colorbar") else False
        if mode not in ("shared", "each", False):
            raise ValueError("colorbar must be 'shared', 'each', or False.")
        cbar_kwargs = options.pop("cbar_kwargs", {})
        label = options.pop("label", None)
        if self._variable_facet and mode != "each":
            units = {
                str(v.attrs.get("units", ""))
                for v in self.original_data.data_vars.values()
            }
            if len(units) > 1:
                raise ValueError(
                    "Variables have incompatible units; use colorbar='each' or convert units first."
                )
        x = x or options.pop("lon_coord", None)
        y = y or options.pop("lat_coord", None)
        x, y = identify_coords(self.data, x, y)
        validate_data_crs(
            self.data, x, y, self.data_crs if self._data_crs_explicit else None
        )
        # Determine the field used for normalization, including model-minus-obs bias.
        if "col1" in options and "col2" in options:
            scale_data = self.data[options["col2"]] - self.data[options["col1"]]
        elif isinstance(self.data, xr.Dataset):
            if var_name is None:
                raise ValueError(
                    "Specify var_name when plotting a Dataset with multiple variables."
                )
            scale_data = self.data[var_name]
        elif isinstance(self.data, xr.DataArray):
            scale_data = self.data
        else:
            if var_name is None:
                raise ValueError("Specify var_name for tabular spatial facets.")
            scale_data = self.data[var_name]
        if mode != "each":
            options = _shared_scale(scale_data, options, plotter_class)
        constructor_parameters = signature(plotter_class.__init__).parameters
        constructor_keys = {
            name
            for name, param in constructor_parameters.items()
            if name not in {"self", "data", "modelvar"}
            and param.kind not in {Parameter.VAR_KEYWORD, Parameter.VAR_POSITIONAL}
        }
        constructor_options = dict(self.plotter_kwargs)
        for key in list(options):
            # Render vmin/vmax/norm/cmap together so a shared norm stays intact.
            if key in constructor_keys and key not in {"vmin", "vmax", "cmap"}:
                constructor_options[key] = options.pop(key)
        feature_keys = {
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
            "natural_earth",
        }
        features = {
            key: options.pop(key) for key in list(options) if key in feature_keys
        }
        features.setdefault("coastlines", True)
        for ax, subset, _ in self._panels():
            if isinstance(subset, xr.Dataset) and var_name is not None:
                subset = subset[var_name]
            elif (
                isinstance(subset, pd.DataFrame)
                and not {"col1", "col2"} <= constructor_options.keys()
            ):
                from .spatial import SpatialTrack

                if issubclass(plotter_class, SpatialTrack):
                    subset = xr.DataArray(
                        subset[var_name].to_numpy(),
                        dims="sample",
                        coords={
                            x: ("sample", subset[x].to_numpy()),
                            y: ("sample", subset[y].to_numpy()),
                        },
                    )
                else:
                    subset = subset.set_index([y, x]).to_xarray()[var_name]
            with style_context(self._style):
                plotter = plotter_class(
                    subset,
                    ax=ax,
                    style=self._style,
                    data_crs=self.data_crs,
                    extent_crs=self.extent_crs,
                    lon_coord=x,
                    lat_coord=y,
                    **features,
                    **constructor_options,
                )
                plotter.plot(**options, add_colorbar=False)
            if hasattr(plotter, "gridliner") and any(
                (
                    plotter.gridliner.top_labels,
                    plotter.gridliner.right_labels,
                    plotter.gridliner.left_labels,
                    plotter.gridliner.bottom_labels,
                )
            ):
                index = next(
                    i for i, candidate in enumerate(self.axes.flat) if candidate is ax
                )
                rows, cols = (
                    self.axes.shape if self.axes.ndim == 2 else (1, self.axes.size)
                )
                row_index, col_index = divmod(index, cols)
                plotter.gridliner.top_labels = False
                plotter.gridliner.right_labels = False
                plotter.gridliner.left_labels = col_index == 0
                plotter.gridliner.bottom_labels = (
                    row_index == rows - 1
                    or index + cols >= sum(1 for _ in self._panels())
                )
            self.plotters.append(plotter)
            if plotter.mappable is not None:
                self.mappables.append(plotter.mappable)
                if mode == "each":
                    panel_label = label
                    if panel_label is None and isinstance(subset, xr.DataArray):
                        panel_label = variable_label(subset)
                    cb = self.fig.colorbar(
                        plotter.mappable,
                        ax=ax,
                        **{"label": panel_label or "", **cbar_kwargs},
                    )
                    self.colorbars.append(cb)
        if self.mappables:
            self.mappable = self.mappables[-1]
        if mode == "shared" and self.mappables:
            if label is None and isinstance(scale_data, xr.DataArray):
                label = variable_label(scale_data)
            cb = self.fig.colorbar(
                self.mappables[-1],
                ax=[ax for ax, _, _ in self._panels()],
                **{"label": label or "", **cbar_kwargs},
            )
            self.colorbars.append(cb)
        self._set_default_titles()
        return self


def _shared_scale(data, options, plotter_class):
    """Resolve one norm and contour levels without materializing all panel values."""
    from .spatial_bias_scatter import SpatialBiasScatterPlot
    from .spatial_contour import SpatialContourPlot
    from .spatial_mesh import SpatialMeshPlot

    options = options.copy()
    norm = options.get("norm")
    cmap = options.get("cmap")
    if isinstance(cmap, tuple):
        options["cmap"], tuple_norm = cmap
        norm = norm or tuple_norm
    vmin, vmax = options.pop("vmin", None), options.pop("vmax", None)
    explicit_limits = vmin is not None or vmax is not None
    levels = options.get("levels")
    if levels is not None and not np.isscalar(levels):
        vmin = levels[0] if vmin is None else vmin
        vmax = levels[-1] if vmax is None else vmax
    if norm is not None and explicit_limits:
        raise ValueError("Supply norm or vmin/vmax, not both.")
    norm = norm or Normalize(vmin, vmax)
    if norm.vmin is None or norm.vmax is None:
        if isinstance(norm, LogNorm):
            data = (
                data.where(data > 0)
                if isinstance(data, xr.DataArray)
                else data[data > 0]
            )
        if isinstance(data, xr.DataArray):
            import dask

            low, high = dask.compute(data.min(skipna=True), data.max(skipna=True))
            low, high = float(low), float(high)
        else:
            low, high = float(data.min()), float(data.max())
        if not np.isfinite([low, high]).all():
            raise ValueError(
                "Cannot determine a color scale from entirely missing/non-finite data; supply explicit limits."
            )
        if issubclass(plotter_class, SpatialBiasScatterPlot):
            low, high = -max(abs(low), abs(high)), max(abs(low), abs(high))
        if norm.vmin is None:
            norm.vmin = low
        if norm.vmax is None:
            norm.vmax = high
    if norm.vmin == norm.vmax:
        delta = abs(norm.vmin) * 0.01 or 0.5
        norm.vmin -= delta
        norm.vmax += delta
    options["norm"] = norm
    if issubclass(plotter_class, SpatialContourPlot) and not issubclass(
        plotter_class, SpatialMeshPlot
    ):
        if levels is None and isinstance(norm, BoundaryNorm):
            options["levels"] = norm.boundaries
        elif levels is None or np.isscalar(levels):
            options["levels"] = np.linspace(
                norm.vmin,
                norm.vmax,
                int(levels) if levels is not None else options.get("ncolors", 10) + 1,
            )
    return options
