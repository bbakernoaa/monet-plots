# src/monet_plots/plots/spatial_contour.py
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cartopy.crs as ccrs
import numpy as np

from ..colorbars import colorbar_index
from .spatial import SpatialPlot

if TYPE_CHECKING:
    import matplotlib.axes


class SpatialContourPlot(SpatialPlot):
    """Create a contour plot on a map with an optional discrete colorbar.

    This plot is useful for visualizing spatial data with continuous values.
    """

    def __init__(
        self,
        modelvar: Any,
        gridobj,
        date=None,
        discrete: bool = True,
        ncolors: int = None,
        dtype: str = "int",
        *args,
        **kwargs,
    ):
        """
        Initialize the plot with data and map projection.

        Args:
            modelvar (np.ndarray, pd.DataFrame, pd.Series, xr.DataArray):
                2D model variable array to contour.
            modelvar (np.ndarray, pd.DataFrame, pd.Series, xr.DataArray):
                2D model variable array to contour.
            gridobj (object): Object with LAT and LON variables.
            date (datetime.datetime): Date/time for the plot title.
            discrete (bool): If True, use a discrete colorbar.
            ncolors (int, optional): Number of discrete colors.
            dtype (str): Data type for colorbar tick labels.
            **kwargs: Keyword arguments passed to SpatialPlot for
                projection and features.
        """
        super().__init__(*args, **kwargs)
        self.modelvar = np.asarray(modelvar)
        self.gridobj = gridobj
        self.date = date
        self.discrete = discrete
        self.ncolors = ncolors
        self.dtype = dtype

    def plot(self, **kwargs: Any) -> matplotlib.axes.Axes:
        """Generate the spatial contour plot."""
        # Draw map features and get remaining kwargs for contourf
        plot_kwargs = self.add_features(**kwargs)

        # Try to handle different gridobj structures
        if hasattr(self.gridobj, "variables"):
            lat_var = self.gridobj.variables["LAT"]
            lon_var = self.gridobj.variables["LON"]

            # Flexible indexing based on dimension count
            if lat_var.ndim == 4:
                lat = lat_var[0, 0, :, :].squeeze()
                lon = lon_var[0, 0, :, :].squeeze()
            elif lat_var.ndim == 3:
                lat = lat_var[0, :, :].squeeze()
                lon = lon_var[0, :, :].squeeze()
            else:
                lat = lat_var.squeeze()
                lon = lon_var.squeeze()
        else:
            # Assume it's already an array or similar
            lat = self.gridobj.LAT
            lon = self.gridobj.LON

        # Data is in lat/lon, so specify transform
        plot_kwargs.setdefault("transform", ccrs.PlateCarree())

        mesh = self.ax.contourf(lon, lat, self.modelvar, **plot_kwargs)

        cmap = plot_kwargs.get("cmap")
        levels = plot_kwargs.get("levels")

        if self.discrete:
            ncolors = self.ncolors
            if ncolors is None and levels is not None:
                ncolors = len(levels) - 1
            c, _ = colorbar_index(
                ncolors,
                cmap,
                minval=levels[0],
                maxval=levels[-1],
                dtype=self.dtype,
                ax=self.ax,
            )
        else:
            self.add_colorbar(mesh)

        if self.date:
            titstring = self.date.strftime("%B %d %Y %H")
            self.ax.set_title(titstring)
        self.fig.tight_layout()
        return self.ax
