# src/monet_plots/plots/wind_barbs.py
from .base import BasePlot
import cartopy.crs as ccrs
import cartopy.feature as cfeature

class WindBarbsPlot(BasePlot):
    """Creates a wind barbs plot.

    This class creates a wind barbs plot on a map.
    """
    def __init__(self, projection=ccrs.PlateCarree(), **kwargs):
        """Initializes the plot with a cartopy projection.

        Args:
            projection (cartopy.crs): The cartopy projection to use.
            **kwargs: Additional keyword arguments to pass to `subplots`.
        """
        super().__init__(subplot_kw={'projection': projection}, **kwargs)
        self.ax.coastlines()
        self.ax.add_feature(cfeature.BORDERS, linestyle=':')
        self.ax.add_feature(cfeature.STATES, linestyle=':')

    def plot(self, u=None, v=None, ws=None, wdir=None, x=None, y=None, gridobj=None, **kwargs):
        """Plots the wind barbs data.

        Args:
            u (numpy.ndarray, optional): U-component (east-west) of wind. If provided, v must also be provided.
            v (numpy.ndarray, optional): V-component (north-south) of wind. If provided, v must also be provided.
            ws (numpy.ndarray, optional): The wind speed data. If provided, wdir must also be provided.
            wdir (numpy.ndarray, optional): The wind direction data. If provided, wdir must also be provided.
            x (numpy.ndarray, optional): X-coordinates. Defaults to None
            y (numpy.ndarray, optional): Y-coordinates. Defaults to None
            gridobj (object, optional): The grid object containing the latitude and longitude data.
            **kwargs: Additional keyword arguments to pass to `barbs`.
        """
        from .. import tools
        
        # Validate inputs
        if u is not None and v is not None:
            # Use u, v components directly
            if x is None or y is None:
                if gridobj is None:
                    raise ValueError("Either x, y coordinates or gridobj must be provided when using u, v components")
                # Extract coordinates from grid object
                lat = gridobj.variables['LAT'][0, 0, :, :].squeeze()
                lon = gridobj.variables['LON'][0, 0, :, :].squeeze()
                x, y = lon, lat
        elif ws is not None and wdir is not None:
            # Convert wind speed and direction to u, v components
            if gridobj is None:
                raise ValueError("gridobj must be provided when using ws, wdir components")
            # Extract coordinates from grid object
            lat = gridobj.variables['LAT'][0, 0, :, :].squeeze()
            lon = gridobj.variables['LON'][0, 0, :, :].squeeze()
            x, y = lon, lat
            u, v = tools.wsdir2uv(ws, wdir)
        else:
            raise ValueError("Either (u, v) or (ws, wdir) must be provided")
        
        if 'transform' not in kwargs:
            kwargs['transform'] = ccrs.PlateCarree()

        # Subsample the data for cleaner visualization
        self.ax.barbs(x[::15, ::15], y[::15, ::15], u[::15, ::15], v[::15, ::15], **kwargs)
