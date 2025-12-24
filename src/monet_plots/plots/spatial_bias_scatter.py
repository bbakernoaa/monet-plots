from scipy.stats import scoreatpercentile as score
import cartopy.crs as ccrs
from .spatial import SpatialPlot
from ..colorbars import colorbar_index
from ..plot_utils import to_dataframe
from typing import Any


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
        projection=ccrs.PlateCarree(),
        vmin: float = None,
        vmax: float = None,
        ncolors: int = 15,
        fact: float = 1.5,
        cmap: str = "RdBu_r",
        *args,
        **kwargs,
    ):
        """
        Initialize the plot with data and map projection.

        Args:
            df (pd.DataFrame, np.ndarray, xr.Dataset, xr.DataArray): DataFrame with 'latitude', 'longitude', and data columns.
            col1 (str): Name of the first column (e.g., observations).
            col2 (str): Name of the second column (e.g., model). Bias is calculated as col2 - col1.
            projection (ccrs.Projection): The cartopy projection for the map.
            vmin (float, optional): Minimum for colorscale.
            vmax (float, optional): Maximum for colorscale.
            ncolors (int): Number of discrete colors.
            fact (float): Scaling factor for point sizes.
            cmap (str or Colormap): Colormap for bias values.
            **kwargs: Additional keyword arguments for plotting, including cartopy features
                      like 'coastlines', 'countries', 'states', 'borders', 'ocean',
                      'land', 'rivers', 'lakes', 'gridlines'. These can be True for default styling or a dict for
                      custom styling.
        """
        super().__init__(*args, projection=projection, **kwargs)
        self.df = to_dataframe(df)
        self.col1 = col1
        self.col2 = col2
        self.vmin = vmin
        self.vmax = vmax
        self.ncolors = ncolors
        self.fact = fact
        self.cmap = cmap

    def plot(self, **kwargs):
        """Generate the spatial bias scatter plot."""
        from numpy import around

        scatter_kwargs = self._draw_features(**kwargs)

        # Ensure we are working with a clean copy with no NaNs in relevant columns
        new = self.df[["latitude", "longitude", self.col1, self.col2]].dropna().copy(deep=True)

        diff = new[self.col2] - new[self.col1]
        top = around(score(diff.abs(), per=95))
        c, cmap = colorbar_index(self.ncolors, self.cmap, minval=top * -1, maxval=top, ax=self.ax)

        c.ax.tick_params(labelsize=13)
        colors = diff
        ss = diff.abs() / top * 100.0
        ss[ss > 300] = 300.0

        self.ax.scatter(
            new.longitude.values,
            new.latitude.values,
            c=colors,
            s=ss,
            vmin=-1.0 * top,
            vmax=top,
            cmap=cmap,
            transform=ccrs.PlateCarree(),  # Tell cartopy the data is in lat/lon
            edgecolors="k",
            linewidths=0.25,
            alpha=0.7,
            **scatter_kwargs,
        )
        return c
