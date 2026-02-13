from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cartopy.crs as ccrs
import numpy as np
import xarray as xr

from ..colorbars import colorbar_index
from ..plot_utils import identify_coords
from .spatial import SpatialPlot

if TYPE_CHECKING:
    import matplotlib.axes


class SpatialImshowPlot(SpatialPlot):
    """Create a basic spatial plot using imshow.

    This plot is useful for visualizing 2D model data on a map.
    It leverages xarray's plotting capabilities when possible.
    """

    def __new__(
        cls,
        modelvar: Any,
        gridobj: Any | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> SpatialImshowPlot | Any:
        """
        Intersects initialization to redirect to SpatialFacetGridPlot if needed.
        """
        if isinstance(modelvar, (xr.DataArray, xr.Dataset)) and (
            "col" in kwargs or "row" in kwargs
        ):
            from .facet_grid import SpatialFacetGridPlot

            kwargs.setdefault("plot_func", "imshow")
            return SpatialFacetGridPlot(modelvar, **kwargs)
        return super().__new__(cls)

    def __init__(
        self,
        modelvar: Any,
        gridobj: Any | None = None,
        plotargs: dict[str, Any] | None = None,
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
        self.plotargs = plotargs or {}
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

        imshow_kwargs.setdefault("cmap", "viridis")
        imshow_kwargs.setdefault("origin", "lower")
        imshow_kwargs.setdefault("transform", ccrs.PlateCarree())

        if isinstance(self.modelvar, xr.DataArray):
            # Use xarray's built-in plotting
            lon_coord, lat_coord = identify_coords(self.modelvar)
            imshow_kwargs.setdefault("x", lon_coord)
            imshow_kwargs.setdefault("y", lat_coord)
            imshow_kwargs.setdefault("ax", self.ax)
            imshow_kwargs.setdefault("add_colorbar", not self.discrete)

            img = self.modelvar.plot.imshow(**imshow_kwargs)
        else:
            # Fallback to manual plotting for non-xarray data
            model_data = np.asarray(self.modelvar)

            if self.gridobj is not None:
                lat = self.gridobj.variables["LAT"][0, 0, :, :].squeeze()
                lon = self.gridobj.variables["LON"][0, 0, :, :].squeeze()
                extent = [lon.min(), lon.max(), lat.min(), lat.max()]
            elif hasattr(self.modelvar, "lat") and hasattr(self.modelvar, "lon"):
                lat = self.modelvar.lat
                lon = self.modelvar.lon
                extent = [lon.min(), lon.max(), lat.min(), lat.max()]
            else:
                extent = imshow_kwargs.get("extent", None)

            img = self.ax.imshow(model_data, extent=extent, **imshow_kwargs)

        if self.discrete:
            # Handle discrete colorbar
            if hasattr(img, "get_clim"):
                vmin, vmax = img.get_clim()
            else:
                vmin, vmax = (
                    imshow_kwargs.get("vmin", np.nanmin(np.asarray(self.modelvar))),
                    imshow_kwargs.get("vmax", np.nanmax(np.asarray(self.modelvar))),
                )

            colorbar_index(
                self.ncolors,
                imshow_kwargs["cmap"],
                minval=vmin,
                maxval=vmax,
                ax=self.ax,
            )
        elif not isinstance(self.modelvar, xr.DataArray) or imshow_kwargs.get(
            "add_colorbar"
        ):
            if not isinstance(self.modelvar, xr.DataArray):
                self.add_colorbar(img)

        return self.ax
