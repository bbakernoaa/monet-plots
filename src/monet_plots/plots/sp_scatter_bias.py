from typing import Any

import matplotlib.pyplot as plt
from scipy.stats import scoreatpercentile as score

from ..plot_utils import _set_outline_patch_alpha, to_dataframe
from .spatial import SpatialPlot


class SpScatterBiasPlot(SpatialPlot):
    """Create a spatial scatter plot showing the bias (difference) between two columns in a DataFrame.

    The point size is scaled by the magnitude of the difference between col2 and col1,
    making larger differences more visually prominent. Differences are capped at 300 units
    for display purposes.
    """

    def __init__(
        self,
        df: Any,
        col1: str,
        col2: str,
        *,
        outline: bool = False,
        tight: bool = True,
        global_map: bool = True,
        cbar_kwargs: dict | None = None,
        val_max: float | None = None,
        val_min: float | None = None,
        **kwargs: Any,
    ):
        """
        Initialize the plot with data and map projection.

        Parameters
        ----------
        df : pd.DataFrame, np.ndarray, xr.Dataset, or xr.DataArray
            The input data containing latitude, longitude, and data columns.
        col1 : str
            Name of the first column (reference value).
        col2 : str
            Name of the second column (comparison value).
        outline : bool, optional
            Whether to show the map outline, by default False.
        tight : bool, optional
            Whether to apply `tight_layout`, by default True.
        global_map : bool, optional
            Whether to set global map boundaries, by default True.
        cbar_kwargs : dict, optional
            Keyword arguments for colorbar customization, by default None.
        val_max : float, optional
            Maximum value for color scaling, by default None.
        val_min : float, optional
            Minimum value for color scaling, by default None.
        **kwargs : Any
            Additional keyword arguments passed to `SpatialPlot` for map
            creation (e.g., `projection`, `extent`, `figsize`, `states`).
        """
        super().__init__(**kwargs)
        self.df = to_dataframe(df)
        self.col1 = col1
        self.col2 = col2
        self.outline = outline
        self.tight = tight
        self.global_map = global_map
        self.cbar_kwargs = cbar_kwargs if cbar_kwargs is not None else {}
        self.val_max = val_max
        self.val_min = val_min

        self.add_features()

    def plot(self, **kwargs: Any) -> "plt.Axes":
        """Generate the spatial scatter bias plot."""
        dfnew = (
            self.df[["latitude", "longitude", self.col1, self.col2]]
            .dropna()
            .copy(deep=True)
        )
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
            self.ax.set_global()
        if self.tight:
            self.fig.tight_layout(pad=0)

        return self.ax
