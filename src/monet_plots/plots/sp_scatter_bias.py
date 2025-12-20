
from scipy.stats import scoreatpercentile as score
from .base import BasePlot
from ..plot_utils import _set_outline_patch_alpha, to_dataframe
from ..mapgen import draw_map
from typing import Any


class SpScatterBiasPlot(BasePlot):
    """Create a spatial scatter plot showing the bias (difference) between two columns in a DataFrame.

    The point size is scaled by the magnitude of the difference between col2 and col1,
    making larger differences more visually prominent. Differences are capped at 300 units
    for display purposes.
    """

    def __init__(self, df: Any, col1: str, col2: str, outline: bool = False, tight: bool = True, global_map: bool = True, map_kwargs: dict = {}, cbar_kwargs: dict = {}, val_max: float = None, val_min: float = None, *args, **kwargs):
        """
        Initialize the plot with data and map projection.

        Args:
            df (pd.DataFrame, np.ndarray, xr.Dataset, xr.DataArray): DataFrame with latitude, longitude, and data columns.
            col1 (str): Name of the first column (reference value).
            col2 (str): Name of the second column (comparison value).
            outline (bool): Whether to show the map outline.
            tight (bool): Whether to apply tight_layout.
            global_map (bool): Whether to set global map boundaries.
            map_kwargs (dict): Keyword arguments for draw_map.
            cbar_kwargs (dict): Keyword arguments for colorbar customization.
            val_max (float, optional): Maximum value for color scaling.
            val_min (float, optional): Minimum value for color scaling.
        """
        super().__init__(*args, **kwargs)
        self.df = to_dataframe(df)
        self.col1 = col1
        self.col2 = col2
        self.outline = outline
        self.tight = tight
        self.global_map = global_map
        self.map_kwargs = map_kwargs
        self.cbar_kwargs = cbar_kwargs
        self.val_max = val_max
        self.val_min = val_min

    def plot(self, **kwargs):
        """Generate the spatial scatter bias plot."""
        if self.ax is None:
            self.ax = draw_map(**self.map_kwargs)

        dfnew = self.df[["latitude", "longitude",
                         self.col1, self.col2]].dropna().copy(deep=True)
        dfnew["sp_diff"] = dfnew[self.col2] - dfnew[self.col1]
        top = score(dfnew["sp_diff"].abs(), per=95)
        if self.val_max is not None:
            top = self.val_max

        dfnew["sp_diff_size"] = dfnew["sp_diff"].abs() / top * 100.0
        dfnew.loc[dfnew["sp_diff_size"] > 300, "sp_diff_size"] = 300.0

        dfnew.plot.scatter(
            x="longitude",
            y="latitude",
            c=dfnew["sp_diff"],
            s=dfnew["sp_diff_size"],
            vmin=-1 * top,
            vmax=top,
            ax=self.ax,
            colorbar=True,
            **kwargs,
        )

        if not self.outline:
            from cartopy.mpl.geoaxes import GeoAxes
            if isinstance(self.ax, GeoAxes):
                _set_outline_patch_alpha(self.ax)
        if self.global_map:
            self.ax.set_xlim([-180, 180])
            self.ax.set_ylim([-90, 90])
        if self.tight:
            self.fig.tight_layout(pad=0)

        return self.ax
