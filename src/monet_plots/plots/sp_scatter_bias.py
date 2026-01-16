from __future__ import annotations

from typing import Any

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import scoreatpercentile as score

from ..plot_utils import _set_outline_patch_alpha, to_dataframe
from .spatial import SpatialPlot


class SpScatterBiasPlot(SpatialPlot):
    """Create a spatial scatter plot showing the bias (difference).

    This class provides a plotting interface where the point size is scaled by
    the magnitude of the difference, making larger differences more visually
    prominent. It inherits from :class:`SpatialPlot` to provide the underlying
    map canvas.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the SpScatterBiasPlot.

        This constructor sets up the map canvas by initializing the parent
        :class:`SpatialPlot`.

        Parameters
        ----------
        **kwargs : Any
            Keyword arguments passed to :class:`SpatialPlot`. These control
            the map projection, figure size, and cartopy features. For example:
            `projection=ccrs.LambertConformal()`, `figsize=(10, 8)`,
            `states=True`, `extent=[-125, -70, 25, 50]`.
        """
        super().__init__(**kwargs)
        self.add_features()

    def plot(
        self,
        df: pd.DataFrame | Any,
        col1: str,
        col2: str,
        *,
        outline: bool = False,
        tight: bool = True,
        global_map: bool = True,
        val_max: float | None = None,
        **kwargs: Any,
    ) -> plt.Axes:
        """Generate the spatial scatter bias plot.

        Calculates the difference between `col2` and `col1`, and plots the
        result as a scatter plot on the map. The point size is scaled by the
        absolute magnitude of the difference, and the color is scaled by the
        signed difference.

        Parameters
        ----------
        df : pd.DataFrame or other array-like
            The input data containing latitude, longitude, and data columns.
            Will be converted to a pandas DataFrame.
        col1 : str
            Name of the first column (reference value, e.g., observations).
        col2 : str
            Name of the second column (comparison value, e.g., model output).
        outline : bool, optional
            Whether to show the map outline, by default False.
        tight : bool, optional
            Whether to apply `tight_layout`, by default True.
        global_map : bool, optional
            Whether to set global map boundaries, by default True.
        val_max : float, optional
            Maximum value for color scaling. If None, it is calculated as the
            95th percentile of the absolute difference. Default is None.
        **kwargs : Any
            Additional keyword arguments passed to
            `pandas.DataFrame.plot.scatter`. A `transform` keyword (e.g.,
            `transform=ccrs.PlateCarree()`) is highly recommended.

        Returns
        -------
        plt.Axes
            The matplotlib axes object containing the plot.
        """
        df_proc = to_dataframe(df)
        dfnew = df_proc[["latitude", "longitude", col1, col2]].dropna().copy(deep=True)
        dfnew["sp_diff"] = dfnew[col2] - dfnew[col1]
        top = score(dfnew["sp_diff"].abs(), per=95)
        if val_max is not None:
            top = val_max

        dfnew["sp_diff_size"] = dfnew["sp_diff"].abs() / top * 100.0
        dfnew.loc[dfnew["sp_diff_size"] > 300, "sp_diff_size"] = 300.0

        # For geospatial plots, a transform is required.
        kwargs.setdefault("transform", ccrs.PlateCarree())

        dfnew.plot.scatter(
            x="longitude",
            y="latitude",
            c="sp_diff",
            s="sp_diff_size",
            vmin=-1 * top,
            vmax=top,
            ax=self.ax,
            colorbar=True,
            **kwargs,
        )

        if not outline:
            from cartopy.mpl.geoaxes import GeoAxes

            if isinstance(self.ax, GeoAxes):
                _set_outline_patch_alpha(self.ax)
        if global_map:
            self.ax.set_global()
        if tight:
            self.fig.tight_layout(pad=0)

        return self.ax

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame | Any,
        col1: str,
        col2: str,
        *,
        map_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> SpScatterBiasPlot:
        """Create a SpScatterBiasPlot from a DataFrame.

        This factory method provides a convenient way to create a scatter bias
        plot directly from a dataframe. It initializes the plot, adds map
        features, and renders the data in a single call.

        Parameters
        ----------
        df : pd.DataFrame or other array-like
            The input data.
        col1 : str
            Name of the first column (reference value).
        col2 : str
            Name of the second column (comparison value).
        map_kwargs : dict, optional
            Keyword arguments passed to the `SpScatterBiasPlot` constructor
            to control map features (e.g., `projection`, `states=True`).
            Default is None.
        **kwargs : Any
            Additional keyword arguments passed to the `plot` method.

        Returns
        -------
        SpScatterBiasPlot
            The instance of the plot class with the data rendered.
        """
        if map_kwargs is None:
            map_kwargs = {}

        plot_instance = cls(**map_kwargs)
        plot_instance.plot(df, col1, col2, **kwargs)
        return plot_instance
