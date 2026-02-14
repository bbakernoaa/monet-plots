from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from scipy.stats import scoreatpercentile as score

from ..colorbars import get_discrete_scale
from ..plot_utils import get_plot_kwargs, to_dataframe
from .spatial import SpatialPlot

if TYPE_CHECKING:
    import matplotlib.axes


class SpatialBiasScatterPlot(SpatialPlot):
    """Create a spatial scatter plot showing bias between model and observations.

    The scatter points are colored by the difference (CMAQ - Obs) and sized
    by the absolute magnitude of this difference, making larger biases more visible.
    """

    def __init__(
        self,
        df: Any,
        col1: str,
        col2: str,
        vmin: float = None,
        vmax: float = None,
        ncolors: int = 15,
        fact: float = 1.5,
        cmap: str = "RdBu_r",
        **kwargs,
    ):
        """
        Initialize the plot with data and map projection.

        Args:
            df (pd.DataFrame, np.ndarray, xr.Dataset, xr.DataArray): DataFrame with 'latitude', 'longitude', and data columns.
            col1 (str): Name of the first column (e.g., observations).
            col2 (str): Name of the second column (e.g., model). Bias is calculated as col2 - col1.
            vmin (float, optional): Minimum for colorscale.
            vmax (float, optional): Maximum for colorscale.
            ncolors (int): Number of discrete colors.
            fact (float): Scaling factor for point sizes.
            cmap (str or Colormap): Colormap for bias values.
            **kwargs: Additional keyword arguments for map creation, passed to
                      :class:`monet_plots.plots.spatial.SpatialPlot`. These
                      include `projection`, `figsize`, `ax`, and cartopy
                      features like `states`, `coastlines`, etc.
        """
        super().__init__(**kwargs)
        self.df = to_dataframe(df)
        self.col1 = col1
        self.col2 = col2
        self.vmin = vmin
        self.vmax = vmax
        self.ncolors = ncolors
        self.fact = fact
        self.cmap = cmap

    def plot(self, **kwargs: Any) -> matplotlib.axes.Axes:
        """Generate the spatial bias scatter plot."""
        from numpy import around

        # Separate feature kwargs from scatter kwargs
        scatter_kwargs = self.add_features(**kwargs)

        # Ensure we are working with a clean copy with no NaNs in relevant columns
        new = (
            self.df[["latitude", "longitude", self.col1, self.col2]]
            .dropna()
            .copy(deep=True)
        )

        diff = new[self.col2] - new[self.col1]
        top = around(score(diff.abs(), per=95))

        # Use new scaling tools
        cmap, norm = get_discrete_scale(
            diff, cmap=self.cmap, n_levels=self.ncolors, vmin=-top, vmax=top
        )

        # Create colorbar
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = self.add_colorbar(mappable, format="%1.2g")
        cbar.ax.tick_params(labelsize=10)

        colors = diff
        ss = diff.abs() / top * 100.0
        ss[ss > 300] = 300.0

        # Prepare scatter kwargs
        final_scatter_kwargs = get_plot_kwargs(
            cmap=cmap,
            norm=norm,
            s=ss,
            c=colors,
            transform=ccrs.PlateCarree(),
            edgecolors="k",
            linewidths=0.25,
            alpha=0.7,
            **scatter_kwargs,
        )

        self.ax.scatter(
            new.longitude.values,
            new.latitude.values,
            **final_scatter_kwargs,
        )
        return self.ax

    def hvplot(self, **kwargs):
        """Generate an interactive spatial bias scatter plot using hvPlot."""
        import hvplot.pandas  # noqa: F401

        new = (
            self.df[["latitude", "longitude", self.col1, self.col2]]
            .dropna()
            .copy(deep=True)
        )
        new["bias"] = new[self.col2] - new[self.col1]
        new["abs_bias"] = new["bias"].abs()

        plot_kwargs = {
            "x": "longitude",
            "y": "latitude",
            "c": "bias",
            "s": "abs_bias",
            "geo": True,
            "kind": "scatter",
            "cmap": self.cmap,
            "hover_cols": [self.col1, self.col2, "bias"],
        }
        plot_kwargs.update(kwargs)

        return new.hvplot(**plot_kwargs)
