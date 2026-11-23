from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle

from ..colorbars import cmap_discretize
from ..plot_utils import _update_history, get_plot_kwargs, normalize_data
from .base import figure_style
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
        super().__init__(**kwargs)
        self.data = normalize_data(data)
        self.col1 = col1
        self.col2 = col2
        self.vmin = vmin
        self.vmax = vmax
        self.ncolors = ncolors
        self.fact = fact
        self.cmap = cmap

        _update_history(self.data, "Initialized monet-plots.SpatialBiasScatterPlot")

    @figure_style
    def plot(self, **kwargs: Any) -> matplotlib.axes.Axes:
        """Generate a static publication-quality spatial bias scatter plot (Track A).

        Parameters
        ----------
        **kwargs : Any
            Keyword arguments passed to `matplotlib.pyplot.scatter`.
            Map features (e.g., `coastlines=True`) can also be passed here.

        Returns
        -------
        matplotlib.axes.Axes
            The matplotlib axes object containing the plot.
        """
        # Separate feature kwargs from scatter kwargs
        scatter_kwargs = self.add_features(**kwargs)

        # Handle different data types for bias calculation
        if isinstance(self.data, (xr.Dataset, xr.DataArray)):
            # Vectorized calculation using Xarray/Dask
            diff = self.data[self.col2] - self.data[self.col1]

            # Identify coordinates
            lat_name = next(
                (
                    c
                    for c in ["latitude", "lat"]
                    if c in self.data.coords
                    or c in self.data.data_vars
                    or c in self.data.dims
                ),
                "lat",
            )
            lon_name = next(
                (
                    c
                    for c in ["longitude", "lon"]
                    if c in self.data.coords
                    or c in self.data.data_vars
                    or c in self.data.dims
                ),
                "lon",
            )

            # Materialize native coordinates before broadcasting. NumPy can
            # present the expanded coordinates as views instead of storing a grid.
            plot_ds = xr.Dataset(
                {
                    "diff": diff,
                    "latitudes": self.data[lat_name],
                    "longitudes": self.data[lon_name],
                }
            ).compute()
            diff, lat, lon = xr.broadcast(
                plot_ds["diff"], plot_ds["latitudes"], plot_ds["longitudes"]
            )
            diff_vals = diff.values
            lat_vals = lat.transpose(*diff.dims).values
            lon_vals = lon.transpose(*diff.dims).values
        else:
            # Fallback for Pandas
            df = self.data
            diff_vals = (df[self.col2] - df[self.col1]).to_numpy(
                dtype=float, na_value=np.nan
            )
            lat_name = next((c for c in ["latitude", "lat"] if c in df.columns), "lat")
            lon_name = next((c for c in ["longitude", "lon"] if c in df.columns), "lon")
            lat_vals = df[lat_name].to_numpy(dtype=float, na_value=np.nan)
            lon_vals = df[lon_name].to_numpy(dtype=float, na_value=np.nan)
        valid = np.isfinite(diff_vals)
        values = diff_vals[valid]
        limits = [abs(limit) for limit in (self.vmin, self.vmax) if limit is not None]
        top = (
            max(limits)
            if limits
            else (float(np.quantile(np.abs(values), 0.95)) if values.size else 0)
        )
        top = top or (float(np.max(np.abs(values))) if values.size else 0) or 1.0
        overflow = (
            "both"
            if np.any(values < -top) and np.any(values > top)
            else "min"
            if np.any(values < -top)
            else "max"
            if np.any(values > top)
            else "neither"
        )

        # Use an odd number of bins so zero has a neutral central color.
        bins = max(3, self.ncolors | 1)
        cmap = cmap_discretize(self.cmap, bins)
        norm = BoundaryNorm(np.linspace(-top, top, bins + 1), cmap.N)

        # Create colorbar with units label
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        _units = getattr(self.data, "attrs", {}).get("units", "")
        if isinstance(self.data, xr.Dataset):
            _units = self.data[self.col2].attrs.get("units", _units)
        label = f"Bias ({self.col2} − {self.col1})"
        if _units:
            label += f" [{_units}]"
        cbar = self.add_colorbar(
            mappable, label=label, extend=overflow, ticks=np.linspace(-top, top, 7)
        )
        cbar.ax.tick_params(labelsize=9)

        with np.errstate(divide="ignore", invalid="ignore"):
            ss = np.clip(np.abs(values) / top * 100.0 * self.fact, 12.0, 300.0)

        # Prepare scatter kwargs
        final_scatter_kwargs = get_plot_kwargs(
            cmap=cmap,
            norm=norm,
            s=ss,
            c=values,
            transform=ccrs.PlateCarree(),
            edgecolors="k",
            linewidths=0.55,
            alpha=0.7,
            **scatter_kwargs,
        )

        scatter = self.ax.scatter(
            lon_vals[valid],
            lat_vals[valid],
            **final_scatter_kwargs,
        )
        if "marker" not in scatter_kwargs:
            markers = ("v", "o", "^")
            shapes = [MarkerStyle(marker) for marker in markers]
            paths = [
                shape.get_path().transformed(shape.get_transform()) for shape in shapes
            ]
            scatter.set_paths(
                [paths[index] for index in (np.sign(values) + 1).astype(int)]
            )
            self.ax.legend(
                handles=[
                    Line2D(
                        [], [], marker=marker, color="0.25", linestyle="", label=name
                    )
                    for marker, name in zip(
                        markers, ("Negative", "Zero", "Positive"), strict=True
                    )
                ]
                + (
                    [
                        Line2D(
                            [],
                            [],
                            marker="x",
                            color="0.35",
                            linestyle="",
                            label="Missing",
                        )
                    ]
                    if np.any(~valid)
                    else []
                ),
                loc="upper left",
            )
        if np.any(~valid):
            self.ax.scatter(
                lon_vals[~valid],
                lat_vals[~valid],
                marker="x",
                color="0.35",
                transform=ccrs.PlateCarree(),
                label="Missing bias",
            )
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

        import pandas as pd

        if isinstance(self.data, pd.DataFrame):
            lat_name = next(
                (c for c in ["latitude", "lat"] if c in self.data.columns), "lat"
            )
            lon_name = next(
                (c for c in ["longitude", "lon"] if c in self.data.columns), "lon"
            )

            ds_plot = self.data.copy()
            ds_plot["bias"] = ds_plot[self.col2] - ds_plot[self.col1]
            plot_target = ds_plot
        else:
            lat_name = next(
                (
                    c
                    for c in ["latitude", "lat"]
                    if c in self.data.coords or c in self.data.dims
                ),
                "lat",
            )
            lon_name = next(
                (
                    c
                    for c in ["longitude", "lon"]
                    if c in self.data.coords or c in self.data.dims
                ),
                "lon",
            )

            ds_plot = self.data.copy()
            ds_plot["bias"] = ds_plot[self.col2] - ds_plot[self.col1]
            _update_history(ds_plot, "Calculated bias for hvplot")
            plot_target = ds_plot

        # Track B defaults
        plot_kwargs = {
            "x": lon_name,
            "y": lat_name,
            "c": "bias",
            "geo": True,
            "rasterize": True,
            "cmap": self.cmap,
        }

        plot_kwargs.update(kwargs)

        return plot_target.hvplot.points(**plot_kwargs)
