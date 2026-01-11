# src/monet_plots/plots/wind_barbs.py

from .spatial import SpatialPlot
from .. import tools
import numpy as np
from ..plot_utils import _squeeze_and_validate_coords
from typing import Any
import cartopy.crs as ccrs


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
        self.ws = np.asarray(ws)
        self.wdir = np.asarray(wdir)
        self.gridobj = gridobj

    def plot(self, **kwargs):
        """Generate the wind barbs plot."""
        barb_kwargs = self.add_features(**kwargs)
        barb_kwargs.setdefault("transform", ccrs.PlateCarree())

        lat = _squeeze_and_validate_coords(self.gridobj.variables["LAT"])
        lon = _squeeze_and_validate_coords(self.gridobj.variables["LON"])
        u, v = tools.wsdir2uv(self.ws, self.wdir)

        # Handle 1D or 2D coordinates
        if lon.ndim == 1 and lat.ndim == 1:
            lon, lat = np.meshgrid(lon, lat)

        # Subsample the data for clarity
        skip = barb_kwargs.pop("skip", 15)
        self.ax.barbs(
            lon[::skip, ::skip],
            lat[::skip, ::skip],
            u[::skip, ::skip],
            v[::skip, ::skip],
            **barb_kwargs,
        )
        return self.ax
