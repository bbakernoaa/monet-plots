from .spatial import SpatialPlot
from ..colorbars import colorbar_index
import numpy as np
from typing import Any
import cartopy.crs as ccrs


class SpatialImshow(SpatialPlot):
    """Create a basic spatial plot using imshow.

    This plot is useful for visualizing 2D model data on a map.
    """

    def __init__(self, modelvar: Any, gridobj, plotargs: dict = {},
                 ncolors: int = 15, discrete: bool = False, *args, **kwargs):
        """
        Initialize the plot with data and map projection.

        Args:
            modelvar (np.ndarray, pd.DataFrame, pd.Series, xr.DataArray): 2D model variable.
            gridobj (object): Object with LAT and LON variables to determine extent.
            plotargs (dict): Arguments for imshow.
            ncolors (int): Number of discrete colors.
            discrete (bool): If True, use a discrete colorbar.
            **kwargs: Keyword arguments passed to SpatialPlot for projection and features.
        """
        super().__init__(*args, **kwargs)
        self.modelvar = np.asarray(modelvar)
        self.gridobj = gridobj
        self.plotargs = plotargs
        self.ncolors = ncolors
        self.discrete = discrete

    def plot(self, **kwargs):
        """Generate the spatial imshow plot."""
        imshow_kwargs = self._draw_features(**kwargs)
        imshow_kwargs.update(self.plotargs)

        lat = self.gridobj.variables["LAT"][0, 0, :, :].squeeze()
        lon = self.gridobj.variables["LON"][0, 0, :, :].squeeze()

        # imshow requires the extent [lon_min, lon_max, lat_min, lat_max]
        extent = [lon.min(), lon.max(), lat.min(), lat.max()]

        imshow_kwargs.setdefault("cmap", "viridis")
        imshow_kwargs.setdefault("origin", "lower")
        imshow_kwargs.setdefault("transform", ccrs.PlateCarree())

        img = self.ax.imshow(self.modelvar, extent=extent, **imshow_kwargs)

        if self.discrete:
            vmin, vmax = img.get_clim()
            c, _ = colorbar_index(
                self.ncolors, imshow_kwargs["cmap"], minval=vmin, maxval=vmax, ax=self.ax)
        else:
            c = self.fig.colorbar(img, ax=self.ax)

        return c
