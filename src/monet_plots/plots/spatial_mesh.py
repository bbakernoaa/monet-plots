"""Spatial plotting for regular, irregular and curvilinear grids."""

from __future__ import annotations

from ..geo import grid_data, variable_label
from ..plot_utils import get_plot_kwargs
from .spatial_contour import SpatialContourPlot


class SpatialMeshPlot(SpatialContourPlot):
    """Render a 2D xarray field using pcolormesh in its explicit data CRS.

    Accepts the same constructor options as SpatialContourPlot, including
    row/col faceting. Coordinates may be independent 1D axes or 2D grids.
    """

    def plot(self, **kwargs):
        """Render the mesh; options include norm, cmap, rasterized and shading."""
        options = {**self._plot_defaults, **kwargs}
        add_colorbar = options.pop("add_colorbar", True)
        cbar_kwargs = options.pop("cbar_kwargs", {})
        field = grid_data(self.modelvar, self.lon_coord, self.lat_coord)
        # Cartopy/Matplotlib can coerce the field more than once. Materialize
        # one panel once, including any lazy coordinates, before that boundary.
        if field.chunks is not None or any(
            coord.chunks is not None for coord in field.coords.values()
        ):
            field = field.compute()
        options = self.add_features(**options)
        options.setdefault("shading", "auto")
        options.setdefault("transform", self.data_crs)
        options.setdefault("cmap", "viridis")
        self.mappable = self.ax.pcolormesh(
            field[self.lon_coord].values,
            field[self.lat_coord].values,
            field.values,
            **get_plot_kwargs(**options),
        )
        if add_colorbar:
            self.add_colorbar(
                self.mappable, **{"label": variable_label(field), **cbar_kwargs}
            )
        if not self.ax.get_title():
            self.ax.set_title(field.attrs.get("long_name", field.name or ""))
        return self.ax
