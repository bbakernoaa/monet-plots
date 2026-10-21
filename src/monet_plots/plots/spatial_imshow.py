from __future__ import annotations

from typing import TYPE_CHECKING, Any

import xarray as xr

from ..colorbars import get_discrete_scale
from ..geo import (
    grid_data,
    identify_coords,
    raster_extent,
    validate_data_crs,
    variable_label,
)
from ..plot_utils import _update_history, get_plot_kwargs, normalize_data
from .spatial import SpatialPlot

if TYPE_CHECKING:
    from matplotlib.axes import Axes


class SpatialImshowPlot(SpatialPlot):
    """Create a basic spatial plot using imshow.

    This class provides an xarray-native interface for visualizing 2D model
    data on a map. It supports both Track A (publication-quality static plots)
    and Track B (interactive exploration).
    """

    def __new__(
        cls,
        data: Any = None,
        gridobj: Any | None = None,
        plotargs: dict[str, Any] | None = None,
        modelvar: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Redirect to SpatialFacetGridPlot if faceting is requested.

        This enables a unified interface for both single-panel and multi-panel
        spatial plots, following Xarray's plotting conventions.

        Parameters
        ----------
        data : Any
            The input data to plot.
        gridobj : Any, optional
            Object with LAT and LON variables, by default None.
        plotargs : dict, optional
            Arguments for imshow, by default None.
        **kwargs : Any
            Additional keyword arguments. If faceting arguments (e.g., `col`,
            `row`, or `col_wrap`) are provided, redirects to `SpatialFacetGridPlot`.

        Returns
        -------
        Any
            An instance of SpatialImshowPlot or SpatialFacetGridPlot.
        """
        from .facet_grid import SpatialFacetGridPlot

        modelvar = data if data is not None else modelvar
        ax = kwargs.get("ax")

        # Aligns with Xarray's trigger for faceting
        facet_kwargs = ["col", "row", "col_wrap"]
        is_faceting = any(kwargs.get(k) is not None for k in facet_kwargs)

        # Redirect to FacetGrid if faceting requested and no existing axes
        if ax is None and is_faceting:
            return SpatialFacetGridPlot(
                modelvar,
                _plotter_class=cls,
                _plotter_kwargs={"gridobj": gridobj, "plotargs": plotargs},
                **kwargs,
            )

        # Also redirect if input is a Dataset with multiple variables
        if (
            ax is None
            and isinstance(modelvar, xr.Dataset)
            and len(modelvar.data_vars) > 1
        ):
            # Default to faceting by variable if not specified
            kwargs.setdefault("col", "variable")
            return SpatialFacetGridPlot(
                modelvar,
                _plotter_class=cls,
                _plotter_kwargs={"gridobj": gridobj, "plotargs": plotargs},
                **kwargs,
            )

        return super().__new__(cls)

    def __init__(
        self,
        data: Any = None,
        gridobj: Any | None = None,
        plotargs: dict[str, Any] | None = None,
        ncolors: int = 15,
        discrete: bool = False,
        col: str | None = None,
        row: str | None = None,
        col_wrap: int | None = None,
        size: float | None = None,
        aspect: float | None = None,
        modelvar: Any = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the spatial imshow plot.

        Parameters
        ----------
        data : Any
            The input data to plot. Preferred format is an xarray DataArray.
        gridobj : Any, optional
            Object with LAT and LON variables to determine extent, by default None.
        plotargs : dict, optional
            Arguments for imshow, by default None.
        ncolors : int, optional
            Number of discrete colors for the discrete colorbar, by default 15.
        discrete : bool, optional
            If True, use a discrete colorbar, by default False.
        col : str, optional
            Dimension name to facet by columns. Aligns with Xarray.
        row : str, optional
            Dimension name to facet by rows. Aligns with Xarray.
        col_wrap : int, optional
            Number of columns before wrapping. Aligns with Xarray.
        size : float, optional
            Height (in inches) of each facet. Aligns with Xarray.
        aspect : float, optional
            Aspect ratio of each facet. Aligns with Xarray.
        **kwargs : Any
            Keyword arguments passed to :class:`SpatialPlot` for map features
            and projection.
        """
        lon_coord = kwargs.pop("lon_coord", None)
        lat_coord = kwargs.pop("lat_coord", None)
        super().__init__(**kwargs)

        # Standardize data to Xarray for consistency and lazy evaluation
        self.modelvar = normalize_data(data if data is not None else modelvar)
        if isinstance(self.modelvar, xr.Dataset) and len(self.modelvar.data_vars) == 1:
            self.modelvar = self.modelvar[list(self.modelvar.data_vars)[0]]

        self.gridobj = gridobj
        self.plotargs = plotargs
        self.ncolors = ncolors
        self.discrete = discrete

        try:
            self.lon_coord, self.lat_coord = identify_coords(
                self.modelvar, lon_coord, lat_coord
            )
        except ValueError:
            if gridobj is None or lon_coord is not None or lat_coord is not None:
                raise
            self.lon_coord, self.lat_coord = "lon", "lat"

        if getattr(self, "gridobj", None) is None:
            validate_data_crs(
                self.modelvar,
                self.lon_coord,
                self.lat_coord,
                self.data_crs if self._data_crs_explicit else None,
            )

        # Ensure coordinates are monotonic for correct plotting
        if (
            isinstance(self.modelvar, xr.DataArray)
            and self.lon_coord in self.modelvar.coords
            and self.lat_coord in self.modelvar.coords
        ):
            self.modelvar = self._ensure_monotonic(
                self.modelvar, self.lon_coord, self.lat_coord
            )

        self.data = self.modelvar
        _update_history(self.modelvar, "Initialized monet-plots.SpatialImshowPlot")

    def plot(self, **kwargs: Any) -> Axes:
        """Render a regular raster with pixel-edge bounds in ``data_crs``.

        ``extent`` controls the map viewport in ``extent_crs`` (geographic by
        default). ``image_extent`` explicitly overrides the raster bounds in
        data coordinates. Use SpatialMeshPlot for irregular/curvilinear grids.
        Set ``add_colorbar=False`` when composing multiple layers.
        """
        options = {**self._plot_defaults, **(self.plotargs or {}), **kwargs}
        add_colorbar = options.pop("add_colorbar", True)
        cbar_kwargs = options.pop("cbar_kwargs", {})
        image_extent = options.pop("image_extent", None)
        if self.gridobj is not None:
            if image_extent is None:
                lon = self.gridobj.variables["LON"]
                lat = self.gridobj.variables["LAT"]
                image_extent = [
                    float(lon.min()),
                    float(lon.max()),
                    float(lat.min()),
                    float(lat.max()),
                ]
            values = self.modelvar.values
        else:
            field = grid_data(self.modelvar, self.lon_coord, self.lat_coord)
            if image_extent is None:
                image_extent = raster_extent(field, self.lon_coord, self.lat_coord)
            values = field.values
        if "extent" not in options and self._requested_extent is None:
            self.ax.set_extent(image_extent, crs=self.data_crs)
        options = self.add_features(**options)
        options.setdefault("origin", "lower")
        options.setdefault("transform", self.data_crs)
        options.setdefault("cmap", "viridis")
        if self.discrete and options.get("norm") is None:
            options["cmap"], options["norm"] = get_discrete_scale(
                values,
                cmap=options["cmap"],
                n_levels=self.ncolors,
                vmin=options.pop("vmin", None),
                vmax=options.pop("vmax", None),
            )
        self.mappable = self.ax.imshow(
            values, extent=image_extent, **get_plot_kwargs(**options)
        )
        if add_colorbar:
            label = (
                variable_label(self.modelvar)
                if isinstance(self.modelvar, xr.DataArray)
                else None
            )
            self.add_colorbar(self.mappable, **{"label": label, **cbar_kwargs})
        if not self.ax.get_title():
            title = self.modelvar.attrs.get("long_name") or getattr(
                self.modelvar, "name", None
            )
            if title:
                self.ax.set_title(title)
        return self.ax

    def hvplot(self, **kwargs: Any) -> Any:
        """Generate an interactive spatial plot using hvPlot (Track B).

        This method leverages Datashader for high-performance rendering of
        large geospatial grids.

        Parameters
        ----------
        **kwargs : Any
            Keyword arguments passed to `hvplot.quadmesh`.
            Common options include `cmap`, `title`, and `alpha`.
            `rasterize=True` is used by default for speed.

        Returns
        -------
        holoviews.core.layout.Layout
            The interactive hvPlot object.
        """
        try:
            import hvplot.xarray  # noqa: F401
        except ImportError:
            raise ImportError(
                "hvplot is required for interactive plotting. Install it with 'pip install hvplot'."
            )

        # Track B defaults
        plot_kwargs = {
            "x": self.lon_coord,
            "y": self.lat_coord,
            "geo": True,
            "crs": self.data_crs,
            "rasterize": True,
            "cmap": "viridis",
        }
        plot_kwargs.update(kwargs)

        return self.modelvar.hvplot.quadmesh(**plot_kwargs)
