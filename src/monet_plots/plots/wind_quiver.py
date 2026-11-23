# src/monet_plots/plots/wind_quiver.py

from typing import Any

import cartopy.crs as ccrs
import numpy as np

from .. import tools
from ..plot_utils import is_lazy
from .base import figure_style
from .spatial import SpatialPlot


class WindQuiverPlot(SpatialPlot):
    """Create a quiver plot of wind vectors on a map.

    This plot shows wind speed and direction using arrows.
    """

    def __init__(self, ws: Any, wdir: Any, gridobj, *args, **kwargs):
        """
        Initialize the plot with data and map projection.

        Args:
            ws (np.ndarray, pd.DataFrame, pd.Series, xr.DataArray): 2D array of wind speeds.
            wdir (np.ndarray, pd.DataFrame, pd.Series, xr.DataArray): 2D array of wind directions.
            gridobj (object): Object with LAT and LON variables.
            **kwargs: Keyword arguments passed to SpatialPlot for projection and features.
        """
        super().__init__(*args, **kwargs)
        self.ws = getattr(ws, "data", ws) if is_lazy(ws) else np.asarray(ws)
        self.wdir = getattr(wdir, "data", wdir) if is_lazy(wdir) else np.asarray(wdir)
        self.gridobj = gridobj

    @figure_style
    def plot(self, *, skip=None, **kwargs):
        """Generate the wind quiver plot. Set skip to override automatic grid thinning."""
        quiver_kwargs = self.add_features(**kwargs)
        quiver_kwargs.setdefault("transform", ccrs.PlateCarree())

        lon, lat, u, v = tools.sample_wind_grid(self.ws, self.wdir, self.gridobj, skip)
        quiver_kwargs.setdefault("pivot", "middle")
        quiver_kwargs.setdefault("width", 0.004)
        quiv = self.ax.quiver(
            lon,
            lat,
            u,
            v,
            **quiver_kwargs,
        )
        return quiv
