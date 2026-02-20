from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cartopy.crs as ccrs
import xarray as xr

from ..colorbars import colorbar_index
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
        modelvar: Any,
        gridobj: Any | None = None,
        plotargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Redirect to SpatialFacetGridPlot if faceting is requested.

        This enables a unified interface for both single-panel and multi-panel
        spatial plots.

        Parameters
        ----------
        modelvar : Any
            The input data to plot.
        gridobj : Any, optional
            Object with LAT and LON variables, by default None.
        plotargs : dict, optional
            Arguments for imshow, by default None.
        **kwargs : Any
            Additional keyword arguments. If `col` or `row` are provided,
            redirects to SpatialFacetGridPlot.

        Returns
        -------
        Any
            An instance of SpatialImshowPlot or SpatialFacetGridPlot.
        """
        from .facet_grid import SpatialFacetGridPlot

        ax = kwargs.get("ax")
        row = kwargs.get("row")
        col = kwargs.get("col")

        # Redirect to FacetGrid if col/row provided and no existing axes
        if ax is None and (row is not None or col is not None):
            return SpatialFacetGridPlot(modelvar, **kwargs)

        # Also redirect if input is a Dataset with multiple variables
        if (
            ax is None
            and isinstance(modelvar, xr.Dataset)
            and len(modelvar.data_vars) > 1
        ):
            # Default to faceting by variable if not specified
            kwargs.setdefault("col", "variable")
            return SpatialFacetGridPlot(modelvar, **kwargs)

        return super().__new__(cls)

    def __init__(
        self,
        modelvar: Any,
        gridobj: Any | None = None,
        plotargs: dict[str, Any] | None = None,
        ncolors: int = 15,
        discrete: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the spatial imshow plot.

        Parameters
        ----------
        modelvar : Any
            The input data to plot. Preferred format is an xarray DataArray.
        gridobj : Any, optional
            Object with LAT and LON variables to determine extent, by default None.
        plotargs : dict, optional
            Arguments for imshow, by default None.
        ncolors : int, optional
            Number of discrete colors for the discrete colorbar, by default 15.
        discrete : bool, optional
            If True, use a discrete colorbar, by default False.
        **kwargs : Any
            Keyword arguments passed to :class:`SpatialPlot` for map features
            and projection.
        """
        # Initialize the map canvas via SpatialPlot
        super().__init__(**kwargs)

        # Standardize data to Xarray for consistency and lazy evaluation
        self.modelvar = normalize_data(modelvar)
        if isinstance(self.modelvar, xr.Dataset) and len(self.modelvar.data_vars) == 1:
            self.modelvar = self.modelvar[list(self.modelvar.data_vars)[0]]

        self.gridobj = gridobj
        self.plotargs = plotargs
        self.ncolors = ncolors
        self.discrete = discrete

        # Aero Protocol: Centralized coordinate identification
        try:
            self.lon_coord, self.lat_coord = self._identify_coords(self.modelvar)
        except ValueError:
            self.lon_coord = kwargs.get("lon_coord", "lon")
            self.lat_coord = kwargs.get("lat_coord", "lat")

        # Ensure coordinates are monotonic for correct plotting
        self.modelvar = self._ensure_monotonic(
            self.modelvar, self.lon_coord, self.lat_coord
        )

        _update_history(self.modelvar, "Initialized monet-plots.SpatialImshowPlot")

    def plot(self, **kwargs: Any) -> Axes:
        """Generate a static publication-quality spatial imshow plot (Track A).

        Parameters
        ----------
        **kwargs : Any
            Keyword arguments passed to `matplotlib.pyplot.imshow`.
            Common options include `cmap`, `vmin`, `vmax`, and `alpha`.
            Map features (e.g., `coastlines=True`) can also be passed here.

        Returns
        -------
        matplotlib.axes.Axes
            The matplotlib axes object containing the plot.
        """
        # Automatically compute extent if not provided
        if "extent" not in kwargs:
            if self.gridobj is not None:
                # Handle legacy gridobj
                try:
                    lat_var = self.gridobj.variables["LAT"]
                    lon_var = self.gridobj.variables["LON"]
                    kwargs["extent"] = [
                        float(lon_var.min()),
                        float(lon_var.max()),
                        float(lat_var.min()),
                        float(lat_var.max()),
                    ]
                except (AttributeError, KeyError):
                    kwargs["extent"] = self._get_extent_from_data(
                        self.modelvar, self.lon_coord, self.lat_coord
                    )
            else:
                kwargs["extent"] = self._get_extent_from_data(
                    self.modelvar, self.lon_coord, self.lat_coord
                )

        # Draw map features and get remaining kwargs for imshow
        imshow_kwargs = self.add_features(**kwargs)

        if self.plotargs:
            imshow_kwargs.update(self.plotargs)

        # Set default imshow settings
        imshow_kwargs.setdefault("cmap", "viridis")
        imshow_kwargs.setdefault("origin", "lower")
        imshow_kwargs.setdefault("transform", ccrs.PlateCarree())

        # Extract extent for imshow [left, right, bottom, top]
        extent = imshow_kwargs.pop("extent", None)

        # Delay computation as much as possible
        # For imshow, we still need concrete values for Track A.
        # But we use xarray's values which handle Dask if properly initialized.
        model_values = self.modelvar.values

        # Handle colormap and normalization
        final_kwargs = get_plot_kwargs(**imshow_kwargs)

        img = self.ax.imshow(model_values, extent=extent, **final_kwargs)

        # Handle colorbar
        if self.discrete:
            vmin, vmax = img.get_clim()
            colorbar_index(
                self.ncolors,
                final_kwargs["cmap"],
                minval=vmin,
                maxval=vmax,
                ax=self.ax,
            )
        else:
            self.add_colorbar(img)

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
        import hvplot.xarray  # noqa: F401

        # Track B defaults
        plot_kwargs = {
            "x": self.lon_coord,
            "y": self.lat_coord,
            "geo": True,
            "rasterize": True,
            "cmap": "viridis",
        }
        plot_kwargs.update(kwargs)

        return self.modelvar.hvplot.quadmesh(**plot_kwargs)
