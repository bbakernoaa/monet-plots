# src/monet_plots/plots/spatial_contour.py
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cartopy.crs as ccrs
import numpy as np
import xarray as xr

from ..colorbars import colorbar_index
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
        modelvar: Any,
        gridobj: Any | None = None,
        date: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """Redirect to SpatialFacetGridPlot if faceting is requested.

        This enables a unified interface for both single-panel and multi-panel
        spatial plots.

        Parameters
        ----------
        modelvar : Any
            The input data to contour.
        gridobj : Any, optional
            Object with LAT and LON variables, by default None.
        date : Any, optional
            Date/time for the plot title, by default None.
        **kwargs : Any
            Additional keyword arguments. If `col` or `row` are provided,
            redirects to SpatialFacetGridPlot.

        Returns
        -------
        Any
            An instance of SpatialContourPlot or SpatialFacetGridPlot.
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
        gridobj: Any,
        date: datetime | None = None,
        discrete: bool = True,
        ncolors: int | None = None,
        dtype: str = "int",
        **kwargs: Any,
    ) -> None:
        """Initialize the spatial contour plot.

        Parameters
        ----------
        modelvar : Any
            The input data to contour. Preferred format is an xarray DataArray.
        gridobj : Any
            Object with LAT and LON variables to determine extent.
        date : datetime, optional
            Date/time for the plot title, by default None.
        discrete : bool, optional
            If True, use a discrete colorbar, by default True.
        ncolors : int, optional
            Number of discrete colors for the colorbar, by default None.
        dtype : str, optional
            Data type for colorbar tick labels, by default "int".
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
        self.date = date
        self.discrete = discrete
        self.ncolors = ncolors
        self.dtype = dtype

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

        _update_history(self.modelvar, "Initialized monet-plots.SpatialContourPlot")

    def plot(self, **kwargs: Any) -> Axes:
        """Generate a static publication-quality spatial contour plot (Track A).

        Parameters
        ----------
        **kwargs : Any
            Keyword arguments passed to `matplotlib.pyplot.contourf`.
            Common options include `cmap`, `levels`, `vmin`, `vmax`, and `alpha`.
            Map features (e.g., `coastlines=True`) can also be passed here.

        Returns
        -------
        matplotlib.axes.Axes
            The matplotlib axes object containing the plot.
        """
        lat = None
        lon = None

        if hasattr(self.gridobj, "variables"):
            # Handle legacy gridobj
            try:
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
            except (AttributeError, KeyError):
                pass
        elif self.gridobj is not None:
            # Assume it's already an array or similar
            try:
                lat = self.gridobj.LAT
                lon = self.gridobj.LON
            except AttributeError:
                pass

        # Automatically compute extent if not provided
        if "extent" not in kwargs:
            if lon is not None and lat is not None:
                kwargs["extent"] = [
                    float(lon.min()),
                    float(lon.max()),
                    float(lat.min()),
                    float(lat.max()),
                ]
            else:
                kwargs["extent"] = self._get_extent_from_data(
                    self.modelvar, self.lon_coord, self.lat_coord
                )

        # Draw map features and get remaining kwargs for contourf
        plot_kwargs = self.add_features(**kwargs)

        # Set default contour settings
        plot_kwargs.setdefault("cmap", "viridis")

        # Data is in lat/lon, so specify transform
        plot_kwargs.setdefault("transform", ccrs.PlateCarree())

        # For coordinates and values, we prefer passing the xarray objects directly.
        # This allows Matplotlib to handle the conversion and maintains
        # parity for Dask-backed arrays.
        if lon is None or lat is None:
            longitude = self.modelvar[self.lon_coord]
            latitude = self.modelvar[self.lat_coord]
        else:
            longitude = lon
            latitude = lat

        # Handle colormap and normalization
        final_kwargs = get_plot_kwargs(**plot_kwargs)

        mesh = self.ax.contourf(longitude, latitude, self.modelvar, **final_kwargs)

        # Handle colorbar
        if self.discrete:
            cmap = final_kwargs.get("cmap")
            levels = final_kwargs.get("levels")
            ncolors = self.ncolors
            if ncolors is None and levels is not None:
                if isinstance(levels, int):
                    ncolors = levels - 1
                    # Use a single compute call for efficiency (Aero Protocol)
                    try:
                        import dask

                        dmin, dmax = dask.compute(
                            self.modelvar.min(), self.modelvar.max()
                        )
                    except (ImportError, AttributeError):
                        dmin, dmax = self.modelvar.min(), self.modelvar.max()

                    # Handle pandas DataFrame where .min() returns a Series
                    try:
                        fmin, fmax = float(dmin), float(dmax)
                    except (TypeError, ValueError):
                        # dmin/dmax could be Series if modelvar was a DataFrame
                        fmin = (
                            float(dmin.min())
                            if hasattr(dmin, "min")
                            else float(np.min(dmin))
                        )
                        fmax = (
                            float(dmax.max())
                            if hasattr(dmax, "max")
                            else float(np.max(dmax))
                        )
                    levels_seq = np.linspace(fmin, fmax, levels)
                else:
                    ncolors = len(levels) - 1
                    levels_seq = levels
            else:
                levels_seq = levels

            if levels_seq is None:
                # Fallback: calculate from data to ensure a discrete colorbar
                # if requested but no levels were provided.
                try:
                    import dask

                    dmin, dmax = dask.compute(self.modelvar.min(), self.modelvar.max())
                except (ImportError, AttributeError):
                    dmin, dmax = self.modelvar.min(), self.modelvar.max()

                # Handle pandas DataFrame where .min() returns a Series
                try:
                    fmin, fmax = float(dmin), float(dmax)
                except (TypeError, ValueError):
                    fmin = (
                        float(dmin.min())
                        if hasattr(dmin, "min")
                        else float(np.min(dmin))
                    )
                    fmax = (
                        float(dmax.max())
                        if hasattr(dmax, "max")
                        else float(np.max(dmax))
                    )
                n_lev = self.ncolors if self.ncolors is not None else 10
                levels_seq = np.linspace(fmin, fmax, n_lev + 1)
                ncolors = n_lev

            if levels_seq is not None:
                colorbar_index(
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
        import hvplot.xarray  # noqa: F401

        # Track B defaults
        plot_kwargs = {
            "x": self.lon_coord,
            "y": self.lat_coord,
            "geo": True,
            "rasterize": True,
            "cmap": "viridis",
            "kind": "contourf",
        }
        plot_kwargs.update(kwargs)

        return self.modelvar.hvplot(**plot_kwargs)
