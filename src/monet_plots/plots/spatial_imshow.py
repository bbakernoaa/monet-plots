from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cartopy.crs as ccrs
import numpy as np

from ..colorbars import colorbar_index
from .spatial import SpatialPlot

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.colorbar


class SpatialImshowPlot(SpatialPlot):
    """Create a basic spatial plot using imshow.

    This plot is useful for visualizing 2D model data on a map.
    """

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
        # Combine kwargs: plot() arguments override constructor-passed plot_kwargs
        imshow_kwargs = self.plot_kwargs.copy()
        imshow_kwargs.update(self.add_features(**kwargs))
        if self.plotargs:
            imshow_kwargs.update(self.plotargs)

        # Default settings
        imshow_kwargs.setdefault("cmap", "viridis")
        imshow_kwargs.setdefault("origin", "lower")
        imshow_kwargs.setdefault("transform", ccrs.PlateCarree())

        # Handle data orientation and coordinate identification if using xarray
        import xarray as xr

        if isinstance(self.modelvar, xr.DataArray):
            try:
                lat_name, lon_name = self._identify_coords(self.modelvar)
                data = self._ensure_monotonic(self.modelvar, lat_name, lon_name)

                # Compute extent from coordinates
                lon = data[lon_name]
                lat = data[lat_name]

                # Ensure we have scalar floats for extent
                lon_min = float(lon.min())
                lon_max = float(lon.max())
                lat_min = float(lat.min())
                lat_max = float(lat.max())
                extent = [lon_min, lon_max, lat_min, lat_max]

                model_data = data.values
            except (ValueError, AttributeError, TypeError):
                # Fallback if coordinate detection fails
                model_data = np.asarray(self.modelvar)
                extent = imshow_kwargs.get("extent", None)
        elif self.gridobj is not None:
            # Traditional gridobj handling
            try:
                lat = self.gridobj.variables["LAT"][0, 0, :, :].squeeze()
                lon = self.gridobj.variables["LON"][0, 0, :, :].squeeze()
                extent = [
                    float(lon.min()),
                    float(lon.max()),
                    float(lat.min()),
                    float(lat.max()),
                ]
            except (KeyError, AttributeError, TypeError):
                extent = imshow_kwargs.get("extent", None)
            model_data = np.asarray(self.modelvar)
        else:
            # Fallback
            model_data = np.asarray(self.modelvar)
            extent = imshow_kwargs.get("extent", None)

        # imshow requires the extent [lon_min, lon_max, lat_min, lat_max]
        if extent is not None:
            imshow_kwargs["extent"] = extent

        # Remove arguments that imshow doesn't support (common in FacetGrid)
        imshow_kwargs.pop("color", None)
        imshow_kwargs.pop("label", None)

        img = self.ax.imshow(model_data, **imshow_kwargs)

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
