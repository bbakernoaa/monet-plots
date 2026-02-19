from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

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

            # Efficient percentile calculation
            try:
                top_val = diff.assign_coords({"abs_diff": np.abs(diff)}).abs_diff.quantile(0.95)
                if hasattr(top_val, "compute"):
                    top = float(top_val.compute())
                else:
                    top = float(top_val)
            except (ImportError, AttributeError, ValueError):
                top = float(np.nanquantile(np.abs(diff.values), 0.95))

            top = np.around(top)

            # Identify coordinates
            lat_name = next((c for c in ["latitude", "lat"] if c in self.data.coords or c in self.data.dims), "lat")
            lon_name = next((c for c in ["longitude", "lon"] if c in self.data.coords or c in self.data.dims), "lon")

            # Compute only what's necessary for plotting
            plot_ds = xr.Dataset({
                "diff": diff,
                "lat": self.data[lat_name],
                "lon": self.data[lon_name]
            })

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
            top = np.around(np.nanquantile(np.abs(diff_vals), 0.95))

        # Use scaling tools
        cmap, norm = get_discrete_scale(
            diff_vals, cmap=self.cmap, n_levels=self.ncolors, vmin=-top, vmax=top
        )

        # Create colorbar
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = self.add_colorbar(mappable, format="%1.2g")
        cbar.ax.tick_params(labelsize=10)

        ss = np.abs(diff_vals) / top * 100.0 * self.fact
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
        import pandas as pd

        if isinstance(self.data, pd.DataFrame):
            import hvplot.pandas  # noqa: F401
            lat_name = next((c for c in ["latitude", "lat"] if c in self.data.columns), "lat")
            lon_name = next((c for c in ["longitude", "lon"] if c in self.data.columns), "lon")

            ds_plot = self.data.copy()
            ds_plot["bias"] = ds_plot[self.col2] - ds_plot[self.col1]
            plot_target = ds_plot
        else:
            import hvplot.xarray  # noqa: F401
            lat_name = next((c for c in ["latitude", "lat"] if c in self.data.coords or c in self.data.dims), "lat")
            lon_name = next((c for c in ["longitude", "lon"] if c in self.data.coords or c in self.data.dims), "lon")

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
