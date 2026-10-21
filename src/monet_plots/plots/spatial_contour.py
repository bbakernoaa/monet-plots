# src/monet_plots/plots/spatial_contour.py
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import xarray as xr

from ..geo import grid_data, identify_coords, validate_data_crs, variable_label
from ..plot_utils import _update_history, get_plot_kwargs, normalize_data
from .spatial import SpatialPlot

if TYPE_CHECKING:
    from datetime import datetime

    from matplotlib.axes import Axes


class SpatialContourPlot(SpatialPlot):
    """Create a contour plot on a map with an optional discrete colorbar.

    This class provides an xarray-native interface for visualizing spatial
    data with continuous values. It supports both Track A (publication-quality
    static plots) and Track B (interactive exploration).
    """

    def __new__(
        cls,
        data: Any = None,
        gridobj: Any | None = None,
        date: Any | None = None,
        modelvar: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Redirect to SpatialFacetGridPlot if faceting is requested.

        This enables a unified interface for both single-panel and multi-panel
        spatial plots, following Xarray's plotting conventions.

        Parameters
        ----------
        data : Any
            The input data to contour.
        gridobj : Any, optional
            Object with LAT and LON variables, by default None.
        date : Any, optional
            Date/time for the plot title, by default None.
        **kwargs : Any
            Additional keyword arguments. If faceting arguments (e.g., `col`,
            `row`, or `col_wrap`) are provided, redirects to `SpatialFacetGridPlot`.

        Returns
        -------
        Any
            An instance of SpatialContourPlot or SpatialFacetGridPlot.
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
                _plotter_kwargs={"gridobj": gridobj, "date": date},
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
                _plotter_kwargs={"gridobj": gridobj, "date": date},
                **kwargs,
            )

        return super().__new__(cls)

    def __init__(
        self,
        data: Any = None,
        gridobj: Any | None = None,
        date: datetime | None = None,
        discrete: bool = True,
        ncolors: int | None = None,
        dtype: str = "int",
        col: str | None = None,
        row: str | None = None,
        col_wrap: int | None = None,
        size: float | None = None,
        aspect: float | None = None,
        modelvar: Any = None,
        **kwargs: Any,
    ) -> None:
        self._is_facetgrid = kwargs.pop("_is_facetgrid", False)
        """Initialize the spatial contour plot.

        Parameters
        ----------
        data : Any
            The input data to contour. Preferred format is an xarray DataArray.
        gridobj : Any, optional
            Object with LAT and LON variables to determine extent, by default None.
        date : datetime, optional
            Date/time for the plot title, by default None.
        discrete : bool, optional
            If True, use a discrete colorbar, by default True.
        ncolors : int, optional
            Number of discrete colors for the colorbar, by default None.
        dtype : str, optional
            Data type for colorbar tick labels, by default "int".
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
        self.date = date
        self.discrete = discrete
        self.ncolors = ncolors
        self.dtype = dtype

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
        _update_history(self.modelvar, "Initialized monet-plots.SpatialContourPlot")

    def plot(self, **kwargs: Any) -> Axes:
        """Render filled contours, retaining the actual contour levels in its colorbar.

        Rendering options can be supplied at construction or here. ``transform``
        defaults to ``data_crs``; ``extent`` sets the viewport in ``extent_crs``.
        """
        options = {**self._plot_defaults, **kwargs}
        add_colorbar = options.pop("add_colorbar", True)
        cbar_kwargs = options.pop("cbar_kwargs", {})
        if self.gridobj is not None:

            def coordinate(name):
                value = self.gridobj.variables[name]
                while value.ndim > 2:
                    value = value[0]
                return value

            longitude, latitude = coordinate("LON"), coordinate("LAT")
            field = self.modelvar
        else:
            field = grid_data(self.modelvar, self.lon_coord, self.lat_coord)
            longitude = field[self.lon_coord]
            latitude = field[self.lat_coord]
        options = self.add_features(**options)
        options.setdefault("cmap", "viridis")
        options.setdefault("transform", self.data_crs)
        if self.ncolors is not None:
            options.setdefault("levels", self.ncolors + 1)
        self.mappable = self.ax.contourf(
            longitude, latitude, field, **get_plot_kwargs(**options)
        )
        if add_colorbar:
            label = variable_label(field) if isinstance(field, xr.DataArray) else None
            self.add_colorbar(self.mappable, **{"label": label, **cbar_kwargs})
        if self.date:
            self.ax.set_title(self.date.strftime("%B %d %Y %H"))
        if not self.ax.get_title():
            title = self.modelvar.attrs.get("long_name") or getattr(
                self.modelvar, "name", None
            )
            if title:
                self.ax.set_title(title)
        return self.ax

    def hvplot(self, **kwargs: Any) -> Any:
        """Generate an interactive spatial contour plot using hvPlot (Track B).

        Parameters
        ----------
        **kwargs : Any
            Keyword arguments passed to `hvplot.contourf`.
            Common options include `cmap`, `levels`, `title`, and `alpha`.

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
            "kind": "contourf",
        }
        plot_kwargs.update(kwargs)

        return self.modelvar.hvplot(**plot_kwargs)
