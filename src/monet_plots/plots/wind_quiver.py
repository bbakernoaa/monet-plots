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
        """Generate the wind quiver plot."""
        quiver_kwargs = self.add_features(**kwargs)
        quiver_kwargs.setdefault("transform", ccrs.PlateCarree())

        lat = self.gridobj.variables["LAT"][0, 0, :, :].squeeze()
        lon = self.gridobj.variables["LON"][0, 0, :, :].squeeze()
        u, v = tools.wsdir2uv(self.ws, self.wdir)
        # Subsample the data for clarity
        quiv = self.ax.quiver(
            lon[::15, ::15],
            lat[::15, ::15],
            u[::15, ::15],
            v[::15, ::15],
            **quiver_kwargs,
        )
        return quiv

    def hvplot(self, **kwargs):
        """Generate an interactive wind quiver plot using hvPlot."""
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
            "title": "Wind Quiver",
        }
        plot_kwargs.update(kwargs)

        return df.hvplot(**plot_kwargs)
