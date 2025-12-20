# src/monet_plots/plots/wind_quiver.py

from .spatial import SpatialPlot
from .. import tools
import numpy as np
from typing import Any
import cartopy.crs as ccrs


class WindQuiverPlot(SpatialPlot):
    """Create a quiver plot of wind vectors on a map.

    This plot shows wind speed and direction using arrows.
    """

    def __init__(self, ws: Any, wdir: Any, gridobj, *args, **kwargs):
        """
        Initialize the plot with data and map projection.

        Args:
            ws (np.ndarray, pd.DataFrame, pd.Series, xr.DataArray): 2D wind speeds.
            wdir (np.ndarray, pd.DataFrame, pd.Series, xr.DataArray): 2D wind directions.
            gridobj (object): Object with LAT and LON variables.
            **kwargs: Keyword arguments for SpatialPlot.
        """
        super().__init__(*args, **kwargs)
        self.ws = np.asarray(ws)
        self.wdir = np.asarray(wdir)
        self.gridobj = gridobj

    def plot(self, **kwargs):
        """Generate the wind quiver plot."""
        quiver_kwargs = self._draw_features(**kwargs)
        quiver_kwargs.setdefault('transform', ccrs.PlateCarree())

        lat = self.gridobj.variables["LAT"][0, 0, :, :].squeeze()
        lon = self.gridobj.variables["LON"][0, 0, :, :].squeeze()
        u, v = tools.wsdir2uv(self.ws, self.wdir)
        # Subsample the data for clarity
        skip = quiver_kwargs.pop('skip', 15)
        quiv = self.ax.quiver(lon[::skip, ::skip], lat[::skip, ::skip],
                              u[::skip, ::skip], v[::skip, ::skip], **quiver_kwargs)
        return quiv
