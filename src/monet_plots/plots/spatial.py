# src/monet_plots/plots/spatial.py
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Union

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from numpy.typing import ArrayLike

from .base import BasePlot

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

# Type hint for array-like data
DataHint = Union[ArrayLike, pd.Series, xr.DataArray]


class SpatialPlot(BasePlot):
    """A base class for creating spatial plots with cartopy.

    This class provides a high-level interface for geospatial plots, handling
    the setup of cartopy axes and the addition of common map features like
    coastlines, states, and gridlines.

    Attributes
    ----------
    resolution : str
        The resolution of the cartopy features (e.g., '50m').
    """

    def __init__(
        self,
        *,
        projection: ccrs.Projection = ccrs.PlateCarree(),
        fig: matplotlib.figure.Figure | None = None,
        ax: matplotlib.axes.Axes | None = None,
        figsize: tuple[float, float] | None = None,
        subplot_kw: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the spatial plot and draw map features.

        This constructor sets up the matplotlib Figure and Cartopy GeoAxes,
        and provides a single interface to draw common map features like
        coastlines and states.

        Parameters
        ----------
        projection : ccrs.Projection, optional
            The Cartopy projection for the map, by default ccrs.PlateCarree().
        fig : matplotlib.figure.Figure, optional
            An existing matplotlib Figure object. If `None`, a new one is
            created.
        ax : matplotlib.axes.Axes, optional
            An existing matplotlib Axes object. If `None`, a new one is
            created.
        figsize : tuple[float, float], optional
            Width, height of the figure in inches.
        subplot_kw : dict[str, Any], optional
            Keyword arguments passed to `fig.add_subplot`. The 'projection' is
            added to these keywords automatically.
        **kwargs : Any
            Keyword arguments for map features, passed to :meth:`add_features`.
            Common options include `coastlines`, `states`, `countries`,
            `ocean`, `land`, `lakes`, `rivers`, `borders`, `gridlines`,
            `extent`, and `resolution`.
        """
        # Ensure 'projection' is correctly passed to subplot creation.
        current_subplot_kw = subplot_kw.copy() if subplot_kw else {}
        current_subplot_kw["projection"] = projection

        self.resolution = kwargs.pop("resolution", "50m")

        # Initialize the base plot, which creates the figure and axes.
        super().__init__(fig=fig, ax=ax, figsize=figsize, subplot_kw=current_subplot_kw)

        # Add features from kwargs
        self.add_features(**kwargs)

    def _get_feature_registry(self, resolution: str) -> dict[str, dict[str, Any]]:
        """Return a registry of Cartopy features and their default styles.

        This approach centralizes feature management, making it easier to
        add new features and maintain existing ones.

        Parameters
        ----------
        resolution : str
            The resolution for the Cartopy features (e.g., '10m', '50m').

        Returns
        -------
        dict[str, dict[str, Any]]
            A dictionary mapping feature names to a specification dictionary
            containing the feature object and its default styling.
        """
        from cartopy.feature import (
            BORDERS,
            COASTLINE,
            LAKES,
            LAND,
            OCEAN,
            RIVERS,
            STATES,
        )

        # Define default styles, falling back to sane defaults if not in rcParams.
        coastline_defaults = {
            "linewidth": plt.rcParams.get("coastline.width", 0.5),
            "edgecolor": plt.rcParams.get("coastline.color", "black"),
            "facecolor": "none",
        }
        states_defaults = {
            "linewidth": plt.rcParams.get("states.width", 0.5),
            "edgecolor": plt.rcParams.get("states.color", "black"),
            "facecolor": "none",
        }
        borders_defaults = {
            "linewidth": plt.rcParams.get("borders.width", 0.5),
            "edgecolor": plt.rcParams.get("borders.color", "black"),
            "facecolor": "none",
        }

        feature_mapping = {
            "coastlines": {
                "feature": COASTLINE.with_scale(resolution),
                "defaults": coastline_defaults,
            },
            "countries": {
                "feature": BORDERS.with_scale(resolution),
                "defaults": borders_defaults,
            },
            "states": {
                "feature": STATES.with_scale(resolution),
                "defaults": states_defaults,
            },
            "borders": {
                "feature": BORDERS.with_scale(resolution),
                "defaults": borders_defaults,
            },
            "ocean": {"feature": OCEAN.with_scale(resolution), "defaults": {}},
            "land": {"feature": LAND.with_scale(resolution), "defaults": {}},
            "rivers": {"feature": RIVERS.with_scale(resolution), "defaults": {}},
            "lakes": {"feature": LAKES.with_scale(resolution), "defaults": {}},
            "counties": {
                "feature": cfeature.NaturalEarthFeature(
                    category="cultural",
                    name="admin_2_counties",
                    scale=resolution,
                    facecolor="none",
                ),
                "defaults": borders_defaults,
            },
        }
        return feature_mapping

    @staticmethod
    def _get_style(
        style: bool | dict[str, Any], defaults: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Get a style dictionary for a map feature.

        Parameters
        ----------
        style : bool | dict[str, Any]
            The user-provided style. If ``True``, use defaults. If a ``dict``,
            use the provided dictionary.
        defaults : dict[str, Any], optional
            The default style to apply if ``style`` is ``True``.

        Returns
        -------
        dict[str, Any]
            The resolved keyword arguments for styling.
        """
        if isinstance(style, dict):
            return style
        if style and defaults:
            # Return a copy to prevent modifying the defaults dictionary in place
            return defaults.copy()
        return {}

    def _draw_single_feature(
        self, style_arg: bool | dict[str, Any], feature_spec: dict[str, Any]
    ) -> None:
        """Draw a single Cartopy feature on the axes.

        Parameters
        ----------
        style_arg : bool | dict[str, Any]
            The user-provided style for the feature. If ``False``, the feature
            is not drawn.
        feature_spec : dict[str, Any]
            A dictionary containing the feature object and default styles.
        """
        if not style_arg:  # Allows for `coastlines=False`
            return

        style_kwargs = self._get_style(style_arg, feature_spec["defaults"])
        feature = feature_spec["feature"]
        self.ax.add_feature(feature, **style_kwargs)

    def add_features(self, **kwargs: Any) -> dict[str, Any]:
        """Add and style Cartopy features on the map axes.

        This method provides a flexible, data-driven interface to add common
        map features. Features can be enabled with a boolean flag (e.g.,
        ``coastlines=True``) or styled with a dictionary of keyword arguments
        (e.g., ``states=dict(linewidth=2, edgecolor='red')``).

        The `extent` keyword is also supported to set the map boundaries.

        Parameters
        ----------
        **kwargs : Any
            Keyword arguments controlling the features to add and their
            styles. Common options include `coastlines`, `states`,
            `countries`, `ocean`, `land`, `lakes`, `rivers`, `borders`,
            and `gridlines`.

        Returns
        -------
        dict[str, Any]
            A dictionary of the keyword arguments that were not used for
            adding features. This can be useful for passing remaining
            arguments to other functions.
        """
        resolution = kwargs.pop("resolution", self.resolution)
        feature_registry = self._get_feature_registry(resolution)

        # If natural_earth is True, enable a standard set of features
        if kwargs.pop("natural_earth", False):
            for feature in ["ocean", "land", "lakes", "rivers"]:
                kwargs.setdefault(feature, True)

        # Main feature-drawing loop
        for key, feature_spec in feature_registry.items():
            if key in kwargs:
                style_arg = kwargs.pop(key)
                self._draw_single_feature(style_arg, feature_spec)

        # Handle gridlines separately, as they are not a 'feature' but an
        # operation on the axes.
        if "gridlines" in kwargs:
            gridline_style = kwargs.pop("gridlines")
            if gridline_style:  # Allows for `gridlines=False`
                gridline_defaults = {
                    "draw_labels": True,
                    "linestyle": "--",
                    "color": "gray",
                }
                gridline_kwargs = self._get_style(gridline_style, gridline_defaults)
                self.ax.gridlines(**gridline_kwargs)

        # Handle extent after features are drawn
        if "extent" in kwargs:
            extent = kwargs.pop("extent")
            if extent is not None:
                self.ax.set_extent(extent)

        return kwargs


class SpatialTrack(SpatialPlot):
    """Plot a trajectory from an :class:`xarray.DataArray` on a map.

    This class provides an xarray-native interface for visualizing paths,
    such as flight trajectories or pollutant tracks, where a variable
    (e.g., altitude, concentration) is plotted along the path.

    It inherits from :class:`SpatialPlot` to provide the underlying map canvas.

    See Also
    --------
    SpatialPlot : The base class for creating spatial plots with Cartopy.

    Examples
    --------
    Create a sample trajectory using ``xarray`` and ``pandas``:

    .. code-block:: python

        import xarray as xr
        import pandas as pd
        import numpy as np
        import cartopy.crs as ccrs

        time = pd.date_range("2024-07-25", periods=100, freq="h")
        lat = np.linspace(30, 40, 100)
        lon = np.linspace(-100, -80, 100)
        altitude = np.linspace(1000, 5000, 100)
        data = xr.DataArray(
            altitude,
            coords={"time": time, "lat": ("time", lat), "lon": ("time", lon)},
            dims=["time"],
            name="altitude",
        )

        # Create a plot on a Lambert Conformal projection
        track_plot = SpatialTrack(
            data,
            projection=ccrs.LambertConformal(),
            states=True,
            figsize=(10, 8)
        )

        # Plot the trajectory and add a colorbar
        scatter = track_plot.plot(
            cmap="viridis",
            s=20,
            edgecolor="k",
            transform=ccrs.PlateCarree()
        )
        track_plot.fig.colorbar(scatter, label="Altitude (m)")

    """

    def __init__(
        self,
        data: xr.DataArray,
        *,
        lon_coord: str = "lon",
        lat_coord: str = "lat",
        **kwargs: Any,
    ):
        """Initialize the :class:`SpatialTrack` plot.

        This constructor validates the input data and sets up the map canvas
        by initializing the parent :class:`SpatialPlot` and adding map features.

        Parameters
        ----------
        data : xr.DataArray
            The input trajectory data. Must be an ``xarray.DataArray`` with
            coordinates for longitude and latitude.
        lon_coord : str, optional
            Name of the longitude coordinate in the DataArray, by default 'lon'.
        lat_coord : str, optional
            Name of the latitude coordinate in the DataArray, by default 'lat'.
        **kwargs : Any
            Keyword arguments passed to :class:`SpatialPlot`. These control
            the map projection, figure size, and Cartopy features. For example:
            ``projection=ccrs.LambertConformal()``, ``figsize=(10, 8)``,
            ``states=True``, ``extent=[-125, -70, 25, 50]``.
        """
        if not isinstance(data, xr.DataArray):
            raise TypeError("Input 'data' must be an xarray.DataArray.")
        if lon_coord not in data.coords:
            raise ValueError(
                f"Longitude coordinate '{lon_coord}' not found in DataArray."
            )
        if lat_coord not in data.coords:
            raise ValueError(
                f"Latitude coordinate '{lat_coord}' not found in DataArray."
            )

        # Initialize the parent SpatialPlot, which creates the map canvas
        # and draws features from the keyword arguments.
        super().__init__(**kwargs)

        # Set data for plotting
        self.data = data
        self.lon_coord = lon_coord
        self.lat_coord = lat_coord

    def plot(self, **kwargs: Any) -> plt.Artist:
        """Plot the trajectory on the map.

        The track is rendered as a scatter plot, where each point is colored
        according to the ``data`` values.

        This method is Dask-aware: if the input data is a Dask array, it will
        compute only the min/max for color scaling, preserving laziness until
        the final rendering.

        Parameters
        ----------
        **kwargs : Any
            Keyword arguments passed to :meth:`matplotlib.axes.Axes.scatter`.
            A ``transform`` keyword (e.g., ``transform=ccrs.PlateCarree()``)
            is highly recommended for geospatial accuracy.
            The ``cmap`` argument can be a string, a Colormap object, or a
            (colormap, norm) tuple from the scaling tools in `colorbars.py`.

        Returns
        -------
        matplotlib.artist.Artist
            The scatter plot artist created by `ax.scatter`.
        """
        from ..plot_utils import get_plot_kwargs

        try:
            import dask.array as da
        except ImportError:
            da = None

        kwargs.setdefault("transform", ccrs.PlateCarree())

        longitude = self.data[self.lon_coord].values
        latitude = self.data[self.lat_coord].values
        c = self.data

        # Aero Protocol: Preserve Laziness
        # If data is a Dask array, compute vmin/vmax now to prevent Matplotlib
        # from triggering a full compute() on the whole array.
        is_dask = da is not None and isinstance(c.data, da.Array)
        if is_dask and "vmin" not in kwargs and "vmax" not in kwargs:
            # cheap metadata-only operation
            vmin, vmax = c.min().compute(), c.max().compute()
            kwargs.setdefault("vmin", vmin)
            kwargs.setdefault("vmax", vmax)

        # Use get_plot_kwargs to handle (cmap, norm) tuples. Pass the raw
        # numpy/dask array to the plotting function.
        final_kwargs = get_plot_kwargs(c=c.data, **kwargs)

        sc = self.ax.scatter(longitude, latitude, **final_kwargs)

        # Add history for provenance
        history = self.data.attrs.get("history", "")
        self.data.attrs["history"] = f"Plotted with monet-plots.SpatialTrack; {history}"

        return sc
