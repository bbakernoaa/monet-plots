# src/monet_plots/plots/wind_barbs.py

from typing import Any

import cartopy.crs as ccrs
import numpy as np

from .. import tools
from ..plot_utils import is_lazy
from .base import figure_style
from .spatial import SpatialPlot


class WindBarbsPlot(SpatialPlot):
    """Create a barbs plot of wind on a map.

    This plot shows wind speed and direction using barbs.
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
        """Generate the wind barbs plot. Set skip to override automatic grid thinning."""
        barb_kwargs = self.add_features(**kwargs)
        barb_kwargs.setdefault("transform", ccrs.PlateCarree())

        lon, lat, u, v = tools.sample_wind_grid(self.ws, self.wdir, self.gridobj, skip)
        barb_kwargs.setdefault("length", 5)
        self.ax.barbs(
            lon,
            lat,
            u,
            v,
            **barb_kwargs,
        )
        return self.ax
