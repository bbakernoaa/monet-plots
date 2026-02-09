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
        gridobj=None,
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
            gridobj (object, optional): Object with LAT and LON variables.
            date (datetime.datetime, optional): Date/time for the plot title.
            discrete (bool): If True, use a discrete colorbar.
            ncolors (int, optional): Number of discrete colors.
            dtype (str): Data type for colorbar tick labels.
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

    def plot(self, **kwargs: Any) -> matplotlib.axes.Axes:
        """Generate the spatial contour plot."""
        # Combine kwargs: plot() arguments override constructor-passed plot_kwargs
        plot_kwargs = self.plot_kwargs.copy()
        plot_kwargs.update(self.add_features(**kwargs))

        import xarray as xr

        # Try to identify coordinates if using xarray
        if isinstance(self.modelvar, xr.DataArray):
            try:
                lat_name, lon_name = self._identify_coords(self.modelvar)
                data = self._ensure_monotonic(self.modelvar, lat_name, lon_name)
                lat = data[lat_name]
                lon = data[lon_name]
                model_data = data.values
                # For contourf, if lat/lon are 1D, we might need meshgrid
                # but matplotlib often handles 1D lat/lon for contourf if they match dims
            except (ValueError, AttributeError):
                model_data = np.asarray(self.modelvar)
                lat, lon = None, None
        else:
            model_data = np.asarray(self.modelvar)
            lat, lon = None, None

        # Try to handle different gridobj structures if xarray failed or not present
        if lat is None or lon is None:
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
            elif hasattr(self.gridobj, "LAT") and hasattr(self.gridobj, "LON"):
                # Assume it's already an array or similar
                lat = self.gridobj.LAT
                lon = self.gridobj.LON

        # Data is in lat/lon, so specify transform
        plot_kwargs.setdefault("transform", ccrs.PlateCarree())

        # Remove arguments that contourf doesn't support (common in FacetGrid)
        plot_kwargs.pop("color", None)
        plot_kwargs.pop("label", None)

        if lat is not None and lon is not None:
            mesh = self.ax.contourf(lon, lat, model_data, **plot_kwargs)
        else:
            mesh = self.ax.contourf(model_data, **plot_kwargs)

        cmap = plot_kwargs.get("cmap")
        levels = plot_kwargs.get("levels")

        if self.discrete:
            ncolors = self.ncolors
            if ncolors is None and levels is not None:
                # If levels is an int, convert to a sequence for len()
                if isinstance(levels, int):
                    ncolors = levels - 1
                    levels_seq = np.linspace(
                        np.nanmin(model_data), np.nanmax(model_data), levels
                    )
                else:
                    ncolors = len(levels) - 1
                    levels_seq = levels
            else:
                levels_seq = levels

            if levels_seq is not None:
                c, _ = colorbar_index(
                    ncolors,
                    cmap,
                    minval=levels_seq[0],
                    maxval=levels_seq[-1],
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
