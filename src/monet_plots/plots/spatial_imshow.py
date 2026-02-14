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
    """Create a basic spatial plot using imshow or pcolormesh.

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

            # Pass plot_func to FacetGrid if provided, otherwise default to imshow
            kwargs.setdefault("plot_func", kwargs.get("plot_func", "imshow"))
            return SpatialFacetGridPlot(modelvar, **kwargs)
        return super().__new__(cls)

    def __init__(
        self,
        modelvar: Any,
        gridobj: Any | None = None,
        plotargs: dict[str, Any] | None = None,
        ncolors: int = 15,
        discrete: bool = False,
        plot_func: str = "imshow",
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
            Arguments for the plotting function.
        ncolors : int, optional
            Number of discrete colors for discrete colorbar.
        discrete : bool, optional
            If True, use a discrete colorbar.
        plot_func : str
            Plotting function to use ('imshow', 'pcolormesh', or 'pcolor').
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
        self.plot_func = plot_func

    def plot(self, **kwargs: Any) -> matplotlib.axes.Axes:
        """Generate the spatial plot.

        Returns
        -------
        matplotlib.axes.Axes
            The axes object containing the plot.
        """
        plot_kwargs = self.add_features(**kwargs)
        if self.plotargs:
            plot_kwargs.update(self.plotargs)

        plot_kwargs.setdefault("cmap", "viridis")
        if self.plot_func == "imshow":
            plot_kwargs.setdefault("origin", "lower")
        plot_kwargs.setdefault("transform", ccrs.PlateCarree())

        if isinstance(self.modelvar, xr.DataArray):
            # Use xarray's built-in plotting
            lon_coord, lat_coord = identify_coords(self.modelvar)
            plot_kwargs.setdefault("x", lon_coord)
            plot_kwargs.setdefault("y", lat_coord)
            plot_kwargs.setdefault("ax", self.ax)
            plot_kwargs.setdefault("add_colorbar", not self.discrete)

            func = getattr(self.modelvar.plot, self.plot_func)
            img = func(**plot_kwargs)
        else:
            # Fallback to manual plotting for non-xarray data
            model_data = np.asarray(self.modelvar)

            func = getattr(self.ax, self.plot_func)

            if self.plot_func == "imshow":
                if self.gridobj is not None:
                    lat = self.gridobj.variables["LAT"][0, 0, :, :].squeeze()
                    lon = self.gridobj.variables["LON"][0, 0, :, :].squeeze()
                    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
                elif hasattr(self.modelvar, "lat") and hasattr(self.modelvar, "lon"):
                    lat = self.modelvar.lat
                    lon = self.modelvar.lon
                    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
                else:
                    extent = plot_kwargs.get("extent", None)
                img = self.ax.imshow(model_data, extent=extent, **plot_kwargs)
            else:
                # pcolormesh, pcolor
                if self.gridobj is not None:
                    lat = self.gridobj.variables["LAT"][0, 0, :, :].squeeze()
                    lon = self.gridobj.variables["LON"][0, 0, :, :].squeeze()
                    img = func(lon, lat, model_data, **plot_kwargs)
                elif hasattr(self.modelvar, "lat") and hasattr(self.modelvar, "lon"):
                    lat = self.modelvar.lat
                    lon = self.modelvar.lon
                    img = func(lon, lat, model_data, **plot_kwargs)
                else:
                    img = func(model_data, **plot_kwargs)

        if self.discrete:
            # Handle discrete colorbar
            if hasattr(img, "get_clim"):
                vmin, vmax = img.get_clim()
            else:
                vmin, vmax = (
                    plot_kwargs.get("vmin", np.nanmin(np.asarray(self.modelvar))),
                    plot_kwargs.get("vmax", np.nanmax(np.asarray(self.modelvar))),
                )

            colorbar_index(
                self.ncolors,
                plot_kwargs["cmap"],
                minval=vmin,
                maxval=vmax,
                ax=self.ax,
            )
        elif not isinstance(self.modelvar, xr.DataArray) or plot_kwargs.get(
            "add_colorbar"
        ):
            if not isinstance(self.modelvar, xr.DataArray):
                self.add_colorbar(img)

        return self.ax

    def hvplot(self, **kwargs):
        """Generate an interactive spatial plot using hvPlot."""
        import hvplot.xarray  # noqa: F401

        if not isinstance(self.modelvar, xr.DataArray):
            raise TypeError("hvplot requires an xarray.DataArray for spatial plots.")

        plot_kwargs = {"geo": True}
        if self.plot_func == "imshow":
            kind = "image"
        else:
            kind = "quadmesh"

        plot_kwargs["kind"] = kind
        plot_kwargs.update(kwargs)

        return self.modelvar.hvplot(**plot_kwargs)
