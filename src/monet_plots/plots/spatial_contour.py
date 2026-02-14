from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Any

import cartopy.crs as ccrs
import numpy as np
import xarray as xr

from ..colorbars import colorbar_index
from ..plot_utils import identify_coords
from .spatial import SpatialPlot

if TYPE_CHECKING:
    import matplotlib.axes


class SpatialContourPlot(SpatialPlot):
    """Create a contour plot on a map with an optional discrete colorbar.

    This plot is useful for visualizing spatial data with continuous values.
    It leverages xarray's plotting capabilities when possible.
    """

    def __new__(
        cls,
        modelvar: Any,
        gridobj: Any | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> SpatialContourPlot | Any:
        """
        Intersects initialization to redirect to SpatialFacetGridPlot if needed.
        """
        if isinstance(modelvar, (xr.DataArray, xr.Dataset)) and (
            "col" in kwargs or "row" in kwargs
        ):
            from .facet_grid import SpatialFacetGridPlot

            # Pass plot_func to FacetGrid if provided, otherwise default to contourf
            kwargs.setdefault("plot_func", kwargs.get("plot_func", "contourf"))
            return SpatialFacetGridPlot(modelvar, **kwargs)
        return super().__new__(cls)

    def __init__(
        self,
        modelvar: Any,
        gridobj: Any | None = None,
        date: datetime.datetime | None = None,
        discrete: bool = True,
        ncolors: int | None = None,
        dtype: str = "int",
        plot_func: str = "contourf",
        *args: Any,
        **kwargs: Any,
    ):
        """
        Initialize the plot with data and map projection.

        Args:
            modelvar (Any): 2D model variable array to contour.
            gridobj (Any): Object with LAT and LON variables.
            date (datetime.datetime, optional): Date/time for the plot title.
            discrete (bool): If True, use a discrete colorbar.
            ncolors (int, optional): Number of discrete colors.
            dtype (str): Data type for colorbar tick labels.
            plot_func (str): Plotting function to use ('contour' or 'contourf').
            *args: Positional arguments for SpatialPlot.
            **kwargs: Keyword arguments passed to SpatialPlot for
                projection and features.
        """
        super().__init__(*args, **kwargs)
        self.modelvar = modelvar
        self.gridobj = gridobj
        self.date = date
        self.discrete = discrete
        self.ncolors = ncolors
        self.dtype = dtype
        self.plot_func = plot_func

    def plot(self, **kwargs: Any) -> matplotlib.axes.Axes:
        """Generate the spatial contour plot."""
        # Draw map features and get remaining kwargs for contour/contourf
        plot_kwargs = self.add_features(**kwargs)

        plot_kwargs.setdefault("transform", ccrs.PlateCarree())

        if isinstance(self.modelvar, xr.DataArray):
            # Use xarray's built-in plotting
            lon_coord, lat_coord = identify_coords(self.modelvar)
            plot_kwargs.setdefault("x", lon_coord)
            plot_kwargs.setdefault("y", lat_coord)
            plot_kwargs.setdefault("ax", self.ax)
            plot_kwargs.setdefault("add_colorbar", not self.discrete)

            func = getattr(self.modelvar.plot, self.plot_func)
            mesh = func(**plot_kwargs)
        else:
            # Fallback to manual plotting
            model_data = np.asarray(self.modelvar)
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
                lat = getattr(self.gridobj, "LAT", None)
                lon = getattr(self.gridobj, "LON", None)

            func = getattr(self.ax, self.plot_func)
            if lat is not None and lon is not None:
                mesh = func(lon, lat, model_data, **plot_kwargs)
            else:
                mesh = func(model_data, **plot_kwargs)

        cmap = plot_kwargs.get("cmap")
        levels = plot_kwargs.get("levels")

        if self.discrete:
            ncolors = self.ncolors
            if ncolors is None and levels is not None:
                # If levels is an int, convert to a sequence for len()
                if isinstance(levels, int):
                    ncolors = levels - 1
                    levels_seq = np.linspace(
                        np.nanmin(np.asarray(self.modelvar)),
                        np.nanmax(np.asarray(self.modelvar)),
                        levels,
                    )
                else:
                    ncolors = len(levels) - 1
                    levels_seq = levels
            else:
                levels_seq = levels

            if levels_seq is not None:
                colorbar_index(
                    ncolors,
                    cmap,
                    minval=levels_seq[0],
                    maxval=levels_seq[-1],
                    dtype=self.dtype,
                    ax=self.ax,
                )
        elif not isinstance(self.modelvar, xr.DataArray) or plot_kwargs.get(
            "add_colorbar"
        ):
            if not isinstance(self.modelvar, xr.DataArray):
                self.add_colorbar(mesh)

        if self.date:
            titstring = self.date.strftime("%B %d %Y %H")
            self.ax.set_title(titstring)
        self.fig.tight_layout()
        return self.ax

    def hvplot(self, **kwargs):
        """Generate an interactive spatial contour plot using hvPlot."""
        import hvplot.xarray  # noqa: F401

        if not isinstance(self.modelvar, xr.DataArray):
            raise TypeError("hvplot requires an xarray.DataArray for spatial plots.")

        plot_kwargs = {"geo": True, "kind": self.plot_func}
        if self.date:
            plot_kwargs["title"] = self.date.strftime("%B %d %Y %H")

        plot_kwargs.update(kwargs)

        return self.modelvar.hvplot(**plot_kwargs)
