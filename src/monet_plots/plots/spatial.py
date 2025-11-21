# src/monet_plots/plots/spatial.py
from .base import BasePlot
from ..colorbars import colorbar_index
import cartopy.crs as ccrs
import cartopy.feature as cfeature

class SpatialPlot(BasePlot):
    """Creates a spatial plot using cartopy.

    This class creates a spatial plot of a 2D model variable on a map.
    It can handle both discrete and continuous colorbars.
    """
    def __init__(self, projection=ccrs.PlateCarree(), fig=None, ax=None, **kwargs):
        """Initializes the plot with a cartopy projection.

        Args:
            projection (cartopy.crs): The cartopy projection to use.
            fig: Pre-existing figure to use (optional)
            ax: Pre-existing axes to use (optional) - if not a GeoAxes, will create a new one
            **kwargs: Additional keyword arguments to pass to `subplots`.
        """
        import cartopy.mpl.geoaxes as geoaxes
        if fig is not None and ax is not None:
            # Check if provided axes is a GeoAxes, if not create a new one at the same position
            if isinstance(ax, geoaxes.GeoAxes):
                # Use provided figure and axes if it's a GeoAxes
                self.fig = fig
                self.ax = ax
            else:
                # Create new GeoAxes at the same position as the provided axes
                # Get the position of the original axes
                pos = ax.get_position()
                self.fig = fig
                self.ax = fig.add_axes([pos.x0, pos.y0, pos.width, pos.height], projection=projection)
                # Remove the original axes and replace it with the new GeoAxes
                # This ensures the test checks the correct axes object
                fig.delaxes(ax)
                # Add the GeoAxes back to the same position
                fig.add_axes(self.ax)
        else:
            # Create new figure and axes with projection
            super().__init__(fig=fig, ax=ax, subplot_kw={'projection': projection}, **kwargs)
        
        # Add cartographic features if axes supports them (i.e., it's a GeoAxes)
        if hasattr(self.ax, 'coastlines'):
            self.ax.add_feature(cfeature.LAND, facecolor='lightgray')
            self.ax.coastlines()
            self.ax.add_feature(cfeature.BORDERS, linestyle=':')
            self.ax.add_feature(cfeature.STATES, linestyle=':')

    def plot(self, modelvar, plotargs={}, ncolors=15, discrete=False, **kwargs):
       """Plots the spatial data.

       Args:
           modelvar (numpy.ndarray): The 2D model variable to plot.
           plotargs (dict, optional): Keyword arguments to pass to `imshow`. Defaults to {}.
           ncolors (int, optional): The number of colors to use for a discrete colorbar. Defaults to 15.
           discrete (bool, optional): Whether to use a discrete colorbar. Defaults to False.
           **kwargs: Additional keyword arguments to pass to `imshow`.
       """
       # Validate discrete parameter type
       if not isinstance(discrete, bool):
           raise TypeError(f"discrete parameter must be boolean, got {type(discrete).__name__}")
       
       # Validate ncolors parameter type and range
       if not isinstance(ncolors, int):
           raise TypeError(f"ncolors parameter must be integer, got {type(ncolors).__name__}")
       if ncolors <= 0 or ncolors > 1000:
           raise ValueError(f"ncolors parameter must be between 1 and 1000, got {ncolors}")
       
       # Validate plotargs parameter type
       if not isinstance(plotargs, dict):
           raise TypeError(f"plotargs parameter must be dict, got {type(plotargs).__name__}")

       if 'cmap' not in plotargs:
           plotargs['cmap'] = 'viridis'

       if 'transform' not in kwargs:
           kwargs['transform'] = ccrs.PlateCarree()

       if discrete:
           vmin = plotargs.get('vmin', modelvar.min())
           vmax = plotargs.get('vmax', modelvar.max())
           # Create discrete colormap without using colorbar_index which causes issues with cartopy
           from matplotlib.colors import BoundaryNorm
           import numpy as np
           bounds = np.linspace(vmin, vmax, ncolors + 1)
           norm = BoundaryNorm(bounds, ncolors)
           plotargs['cmap'] = plotargs['cmap'] if isinstance(plotargs['cmap'], str) else plotargs['cmap'].name
           im = self.ax.imshow(modelvar, norm=norm, **plotargs, **kwargs)
           self.cbar = self.fig.colorbar(im, ax=self.ax, ticks=np.linspace(vmin, vmax, ncolors))
       else:
           im = self.ax.imshow(modelvar, **plotargs, **kwargs)
           self.cbar = self.fig.colorbar(im, ax=self.ax)

       return self.ax
