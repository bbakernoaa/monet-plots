from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import colors as mcolors

from ..colorbars import get_discrete_scale
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
        discrete: bool = False,
        cbar_label: str | None = None,
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
        self.discrete = discrete
        self.cbar_label = cbar_label

        _update_history(self.data, "Initialized monet-plots.SpatialBiasScatterPlot")

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

            # Compute only what's necessary for plotting
            plot_ds = xr.Dataset(
                {"diff": diff, "lat": self.data[lat_name], "lon": self.data[lon_name]}
            )

            # Drop NaNs before compute to minimize transfer
            if plot_ds.dims:
                plot_ds = plot_ds.dropna(dim=list(plot_ds.dims)[0])

            concrete = plot_ds.compute()
            diff_vals = concrete["diff"].values
            lat_vals = concrete["lat"].values
            lon_vals = concrete["lon"].values
        else:
            # Fallback for Pandas
            df = self.data.dropna(subset=[self.col1, self.col2])
            diff_vals = (df[self.col2] - df[self.col1]).values
            lat_name = next((c for c in ["latitude", "lat"] if c in df.columns), "lat")
            lon_name = next((c for c in ["longitude", "lon"] if c in df.columns), "lon")
            lat_vals = df[lat_name].values
            lon_vals = df[lon_name].values

        # Use constructor-provided limits when present; quantiles are fallback only.
        try:
            quantile_top = float(np.around(np.nanquantile(np.abs(diff_vals), 0.95)))
        except (ValueError, TypeError):
            quantile_top = np.nan

        if not np.isfinite(quantile_top) or quantile_top <= 0:
            finite_abs = np.abs(diff_vals[np.isfinite(diff_vals)])
            quantile_top = float(np.nanmax(finite_abs)) if finite_abs.size else 1.0

        if quantile_top <= 0:
            quantile_top = 1.0

        default_vmin = -quantile_top
        default_vmax = quantile_top
        color_vmin = self.vmin if self.vmin is not None else default_vmin
        color_vmax = self.vmax if self.vmax is not None else default_vmax

        size_ref = max(abs(color_vmin), abs(color_vmax))
        if not np.isfinite(size_ref) or size_ref <= 0:
            size_ref = quantile_top
        if size_ref <= 0:
            size_ref = 1.0

        # Use discrete or continuous normalization depending on user preference.
        if self.discrete:
            cmap, norm = get_discrete_scale(
                diff_vals,
                cmap=self.cmap,
                n_levels=self.ncolors,
                vmin=color_vmin,
                vmax=color_vmax,
            )
        else:
            cmap = plt.get_cmap(self.cmap) if isinstance(self.cmap, str) else self.cmap
            norm = mcolors.Normalize(vmin=color_vmin, vmax=color_vmax)

        # Create colorbar with units label
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        _units = getattr(self.data, "attrs", {}).get("units", "")
        _cbar_label = self.cbar_label or (f"Bias ({_units})" if _units else "Bias")
        cbar = self.add_colorbar(mappable, label=_cbar_label)
        cbar.ax.tick_params(labelsize=9)

        with np.errstate(divide="ignore", invalid="ignore"):
            ss = np.abs(diff_vals) / size_ref * 100.0 * self.fact
            ss[np.isnan(ss)] = 0.0
            ss[ss > 300] = 300.0

        # Prepare scatter kwargs
        final_scatter_kwargs = get_plot_kwargs(
            cmap=cmap,
            norm=norm,
            s=ss,
            c=diff_vals,
            transform=ccrs.PlateCarree(),
            edgecolors="k",
            linewidths=0.25,
            alpha=0.7,
            **scatter_kwargs,
        )

        self.ax.scatter(
            lon_vals,
            lat_vals,
            **final_scatter_kwargs,
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
