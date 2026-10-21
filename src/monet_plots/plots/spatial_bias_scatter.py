from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr

from ..colorbars import get_discrete_scale
from ..geo import identify_coords, validate_data_crs
from ..plot_utils import _update_history, get_plot_kwargs, normalize_data
from .spatial import SpatialPlot

if TYPE_CHECKING:
    import matplotlib.axes


class SpatialBiasScatterPlot(SpatialPlot):
    """Create a spatial scatter plot showing bias between model and observations.

    The scatter points are colored by the difference (model - observations) and
    sized by the absolute magnitude of this difference, making larger biases
    more visible. This class supports both Track A (publication) and
    Track B (interactive) visualization.
    """

    def __new__(
        cls,
        data: Any,
        col1: str,
        col2: str,
        **kwargs: Any,
    ) -> Any:
        """Redirect to SpatialFacetGridPlot if faceting is requested."""
        from .facet_grid import SpatialFacetGridPlot

        ax = kwargs.get("ax")
        facet_kwargs = ["col", "row", "col_wrap"]
        is_faceting = any(kwargs.get(k) is not None for k in facet_kwargs)

        if ax is None and is_faceting:
            return SpatialFacetGridPlot(
                data, _plotter_class=cls, col1=col1, col2=col2, **kwargs
            )

        return super().__new__(cls)

    def __init__(
        self,
        data: Any,
        col1: str,
        col2: str,
        vmin: float | None = None,
        vmax: float | None = None,
        ncolors: int = 15,
        fact: float = 1.5,
        cmap: str = "RdBu_r",
        **kwargs: Any,
    ) -> None:
        """Initialize the plot with data and map projection.

        Parameters
        ----------
        data : Any
            Input data. Preferred format is xarray.Dataset or xarray.DataArray
            with 'latitude' and 'longitude' (or 'lat' and 'lon') coordinates.
        col1 : str
            Name of the first variable (e.g., observations).
        col2 : str
            Name of the second variable (e.g., model). Bias is calculated
            as col2 - col1.
        vmin : float, optional
            Minimum for colorscale, by default None.
        vmax : float, optional
            Maximum for colorscale, by default None.
        ncolors : int, optional
            Number of discrete colors, by default 15.
        fact : float, optional
            Scaling factor for point sizes, by default 1.5.
        cmap : str, optional
            Colormap for bias values, by default "RdBu_r".
        **kwargs : Any
            Additional keyword arguments for map creation, passed to
            :class:`monet_plots.plots.spatial.SpatialPlot`.
        """
        lon_coord = kwargs.pop("lon_coord", None)
        lat_coord = kwargs.pop("lat_coord", None)
        super().__init__(**kwargs)
        self.data = normalize_data(data)
        self.col1 = col1
        self.col2 = col2
        self.vmin = vmin
        self.vmax = vmax
        self.ncolors = ncolors
        self.fact = fact
        self.cmap = cmap

        # Identify coordinates
        self.lon_coord, self.lat_coord = identify_coords(
            self.data, lon_coord, lat_coord
        )

        if getattr(self, "gridobj", None) is None:
            validate_data_crs(
                self.data,
                self.lon_coord,
                self.lat_coord,
                self.data_crs if self._data_crs_explicit else None,
            )

        # Ensure coordinates are monotonic if it's a grid (not strictly necessary for scatter but good practice)
        self.data = self._ensure_monotonic(self.data, self.lon_coord, self.lat_coord)

        _update_history(self.data, "Initialized monet-plots.SpatialBiasScatterPlot")

    def plot(self, **kwargs: Any) -> matplotlib.axes.Axes:
        """Plot paired biases with optional shared normalization and colorbar.

        All-zero differences retain a finite, symmetric scale. Gridded
        coordinates are broadcast to the field; missing pairs remain masked.
        """
        options = {**self._plot_defaults, **kwargs}
        add_colorbar = options.pop("add_colorbar", True)
        cbar_kwargs = options.pop("cbar_kwargs", {})
        options = self.add_features(**options)
        diff = self.data[self.col2] - self.data[self.col1]
        if isinstance(diff, xr.DataArray):
            diff, lon, lat = xr.broadcast(
                diff, self.data[self.lon_coord], self.data[self.lat_coord]
            )
            concrete = xr.Dataset({"diff": diff, "lon": lon, "lat": lat}).compute()
            values, longitude, latitude = (
                concrete[k].values.ravel() for k in ("diff", "lon", "lat")
            )
        else:
            values = diff.to_numpy()
            longitude, latitude = self.data[self.lon_coord], self.data[self.lat_coord]
        finite = np.asarray(values)[np.isfinite(values)]
        top = float(np.quantile(np.abs(finite), 0.95)) if finite.size else 1.0
        top = top or 1.0
        norm = options.pop("norm", None)
        cmap = options.pop("cmap", self.cmap)
        if norm is None:
            vmin = options.pop("vmin", self.vmin)
            vmax = options.pop("vmax", self.vmax)
            cmap, norm = get_discrete_scale(
                values,
                cmap=cmap,
                n_levels=self.ncolors,
                vmin=-top if vmin is None else vmin,
                vmax=top if vmax is None else vmax,
            )
        options.setdefault("s", np.minimum(np.abs(values) / top * 100 * self.fact, 300))
        options.setdefault("transform", self.data_crs)
        if not {"edgecolors", "edgecolor", "ec"} & options.keys():
            options["edgecolors"] = "k"
        if not {"linewidths", "linewidth", "lw"} & options.keys():
            options["linewidths"] = 0.5
        self.mappable = self.ax.scatter(
            longitude,
            latitude,
            c=values,
            **get_plot_kwargs(cmap=cmap, norm=norm, **options),
        )
        if add_colorbar:
            units = self.data.attrs.get("units", "")
            label = f"Bias ({units})" if units else "Bias"
            self.add_colorbar(self.mappable, **{"label": label, **cbar_kwargs})
        return self.ax

    def hvplot(self, **kwargs: Any) -> Any:
        """Generate an interactive spatial bias scatter plot using hvPlot (Track B).

        Parameters
        ----------
        **kwargs : Any
            Keyword arguments passed to `hvplot.points`.
            Common options include `cmap`, `title`, and `alpha`.
            `rasterize=True` is used by default for high performance.

        Returns
        -------
        holoviews.core.layout.Layout
            The interactive hvPlot object.
        """
        try:
            import hvplot.pandas  # noqa: F401
            import hvplot.xarray  # noqa: F401
        except ImportError:
            raise ImportError(
                "hvplot is required for interactive plotting. Install it with 'pip install hvplot'."
            )

        ds_plot = self.data.copy()
        ds_plot["bias"] = ds_plot[self.col2] - ds_plot[self.col1]

        if hasattr(ds_plot, "attrs"):
            _update_history(ds_plot, "Calculated bias for hvplot")

        # Track B defaults
        plot_kwargs = {
            "x": self.lon_coord,
            "y": self.lat_coord,
            "c": "bias",
            "geo": True,
            "crs": self.data_crs,
            "rasterize": True,
            "cmap": self.cmap,
        }

        plot_kwargs.update(kwargs)

        return ds_plot.hvplot.points(**plot_kwargs)
