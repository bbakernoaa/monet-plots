# src/monet_plots/plots/wind_quiver.py

from typing import Any

import cartopy.crs as ccrs
import numpy as np

from .. import tools
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
        self.ws = np.asarray(ws)
        self.wdir = np.asarray(wdir)
        self.gridobj = gridobj

    @figure_style
    def plot(self, *, skip=None, **kwargs):
        """Generate the wind quiver plot. Set skip to override automatic grid thinning."""
        quiver_kwargs = self.add_features(**kwargs)
        quiver_kwargs.setdefault("transform", ccrs.PlateCarree())

        lat = self.gridobj.variables["LAT"][0, 0, :, :]
        lon = self.gridobj.variables["LON"][0, 0, :, :]
        u, v = tools.wsdir2uv(self.ws, self.wdir)
        selection = tools.wind_grid_slice(u.shape, skip)
        quiver_kwargs.setdefault("pivot", "middle")
        quiver_kwargs.setdefault("width", 0.004)
        quiv = self.ax.quiver(
            lon[selection],
            lat[selection],
            u[selection],
            v[selection],
            **quiver_kwargs,
        )
        return quiv
