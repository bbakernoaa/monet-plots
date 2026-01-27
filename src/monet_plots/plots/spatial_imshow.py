from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cartopy.crs as ccrs
import numpy as np

from ..colorbars import colorbar_index
import numpy as np
from typing import Any
import cartopy.crs as ccrs


class SpatialImshow(SpatialPlot):
    """Create a basic spatial plot using imshow.

    This plot is useful for visualizing 2D model data on a map.
    """

    def __init__(
        self,
        modelvar: Any,
        gridobj,
        plotargs: dict = {},
        ncolors: int = 15,
        discrete: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the plot with data and map projection.

        Parameters
        ----------
        modelvar : xarray.DataArray or array-like
            2D model variable array to plot.
        gridobj : Any, optional
            Object with LAT and LON variables to determine extent.
        plotargs : dict, optional
            Arguments for imshow.
        ncolors : int, optional
            Number of discrete colors for discrete colorbar.
        discrete : bool, optional
            If True, use a discrete colorbar.
        *args : Any
            Positional arguments for SpatialPlot.
        **kwargs : Any
            Keyword arguments passed to SpatialPlot for projection and features.
        """
        super().__init__(*args, **kwargs)
        self.modelvar = modelvar
        self.gridobj = gridobj
        self.plotargs = plotargs
        self.ncolors = ncolors
        self.discrete = discrete

    def plot(self, **kwargs: Any) -> matplotlib.axes.Axes:
        """Generate the spatial imshow plot.

        Returns
        -------
        matplotlib.axes.Axes
            The axes object containing the plot.
        """
        imshow_kwargs = self.add_features(**kwargs)
        if self.plotargs:
            imshow_kwargs.update(self.plotargs)

        lat = self.gridobj.variables["LAT"][0, 0, :, :].squeeze()
        lon = self.gridobj.variables["LON"][0, 0, :, :].squeeze()

        # imshow requires the extent [lon_min, lon_max, lat_min, lat_max]

        imshow_kwargs.setdefault("cmap", "viridis")
        imshow_kwargs.setdefault("origin", "lower")
        imshow_kwargs.setdefault("transform", ccrs.PlateCarree())

        img = self.ax.imshow(model_data, extent=extent, **imshow_kwargs)

        if self.discrete:
            vmin, vmax = img.get_clim()
            colorbar_index(
                self.ncolors,
                imshow_kwargs["cmap"],
                minval=vmin,
                maxval=vmax,
                ax=self.ax,
            )
        else:
            self.add_colorbar(img)

        return self.ax
