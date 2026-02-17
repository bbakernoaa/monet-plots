# src/monet_plots/plots/wind_barbs.py

from .spatial import SpatialPlot
from .. import tools
import numpy as np
from typing import Any
import cartopy.crs as ccrs


class WindBarbsPlot(SpatialPlot):
    """Create a barbs plot of wind on a map.

    This plot shows wind speed and direction using barbs.
    """

    def __init__(self, ws: Any, wdir: Any, gridobj, *args, fig=None, ax=None, **kwargs):
        """
        Initialize the plot with data and map projection.

        Args:
            ws (np.ndarray, pd.DataFrame, pd.Series, xr.DataArray): 2D array of wind speeds.
            wdir (np.ndarray, pd.DataFrame, pd.Series, xr.DataArray): 2D array of wind directions.
            gridobj (object): Object with LAT and LON variables.
            **kwargs: Keyword arguments passed to SpatialPlot for projection and features.
        """
        super().__init__(fig=fig, ax=ax, **kwargs)
        self.ws = np.asarray(ws)
        self.wdir = np.asarray(wdir)
        self.gridobj = gridobj

    def plot(self, **kwargs):
        """Generate the wind barbs plot."""
        barb_kwargs = self.add_features(**kwargs)
        barb_kwargs.setdefault("transform", ccrs.PlateCarree())

        lat = self.gridobj.variables["LAT"][0, 0, :, :].squeeze()
        lon = self.gridobj.variables["LON"][0, 0, :, :].squeeze()
        u, v = tools.wsdir2uv(self.ws, self.wdir)
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

    def hvplot(self, **kwargs):
        """Generate an interactive wind barbs plot using hvPlot."""
        import hvplot.pandas  # noqa: F401
        import pandas as pd

        lat = self.gridobj.variables["LAT"][0, 0, :, :].squeeze()
        lon = self.gridobj.variables["LON"][0, 0, :, :].squeeze()
        u, v = tools.wsdir2uv(self.ws, self.wdir)

        # Flatten and create DataFrame for hvplot
        skip = kwargs.get("skip", 15)
        df = pd.DataFrame(
            {
                "lon": lon[::skip, ::skip].flatten(),
                "lat": lat[::skip, ::skip].flatten(),
                "u": u[::skip, ::skip].flatten(),
                "v": v[::skip, ::skip].flatten(),
                "ws": self.ws[::skip, ::skip].flatten(),
            }
        ).dropna()

        # Calculate angle and magnitude for hvplot vectorfield
        # angle is radians counter-clockwise from East
        df["angle"] = np.arctan2(df["v"], df["u"])

        plot_kwargs = {
            "x": "lon",
            "y": "lat",
            "angle": "angle",
            "mag": "ws",
            "geo": True,
            "kind": "vectorfield",
            "title": "Wind Barbs (Simplified)",
        }
        plot_kwargs.update(kwargs)

        return df.hvplot(**plot_kwargs)
