# src/monet_plots/plots/spatial.py
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Union

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from numpy.typing import ArrayLike

from ..plot_utils import _update_history
from ..style import get_style_setting
from .base import BasePlot

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

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
        fig: Figure | None = None,
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
        subplot_kw: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the spatial plot and draw map features.

        This constructor sets up the matplotlib Figure and cartopy GeoAxes,
        and provides a single interface to draw common map features like
        coastlines and states.

        Parameters
        ----------
        projection : ccrs.Projection, optional
            The cartopy projection for the map, by default ccrs.PlateCarree().
        fig : plt.Figure | None, optional
            An existing matplotlib Figure object. If None, a new one is
            created, by default None.
        ax : plt.Axes | None, optional
            An existing matplotlib Axes object. If None, a new one is created,
            by default None.
        figsize : tuple[float, float] | None, optional
             Width, height in inches. If not provided, the matplotlib default
             will be used.
        subplot_kw : dict[str, Any] | None, optional
            Keyword arguments passed to `fig.add_subplot`, by default None.
            The 'projection' is added to these keywords automatically.
        **kwargs : Any
            Keyword arguments for map features, passed to `add_features`.
            Common options include `coastlines`, `states`, `countries`,
            `ocean`, `land`, `lakes`, `rivers`, `borders`, `gridlines`,
            `extent`, and `resolution`.

        Attributes
        ----------
        fig : plt.Figure
            The matplotlib Figure object.
        ax : plt.Axes
            The matplotlib Axes (or GeoAxes) object.
        resolution : str
            The default resolution for cartopy features.
        """
        # Ensure 'projection' is correctly passed to subplot creation.
        current_subplot_kw = subplot_kw.copy() if subplot_kw else {}
        current_subplot_kw["projection"] = projection

        self.resolution = kwargs.pop("resolution", "50m")
        style = kwargs.pop("style", "wiley")

        # Ensure coastlines are enabled by default if not specified.
        if "coastlines" not in kwargs:
            kwargs["coastlines"] = True

        # Initialize the base plot, which creates the figure and axes.
        super().__init__(
            fig=fig, ax=ax, figsize=figsize, style=style, subplot_kw=current_subplot_kw
        )

        # If BasePlot didn't create an axes (e.g. because fig was provided),
        # create one now.
        if self.ax is None:
            self.ax = self.fig.add_subplot(1, 1, 1, **current_subplot_kw)

        # Add features from kwargs
        self.add_features(**kwargs)

    def _get_feature_registry(self, resolution: str) -> dict[str, dict[str, Any]]:
        """Return a registry of cartopy features and their default styles.

        This approach centralizes feature management, making it easier to
        add new features and maintain existing ones.

        Parameters
        ----------
        resolution : str
            The resolution for the cartopy features (e.g., '10m', '50m').

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
            "linewidth": get_style_setting("coastline.width", 0.5),
            "edgecolor": get_style_setting("coastline.color", "black"),
            "facecolor": "none",
        }
        states_defaults = {
            "linewidth": get_style_setting("states.width", 0.5),
            "edgecolor": get_style_setting("states.color", "black"),
            "facecolor": "none",
        }
        borders_defaults = {
            "linewidth": get_style_setting("borders.width", 0.5),
            "edgecolor": get_style_setting("borders.color", "black"),
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
        """Get a style dictionary for a feature.

        Parameters
        ----------
        style : bool or dict[str, Any]
            The user-provided style. If True, use defaults. If a dict, use it.
        defaults : dict[str, Any], optional
            The default style to apply if `style` is True.

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
        """Draw a single cartopy feature on the axes.

        Parameters
        ----------
        style_arg : bool or dict[str, Any]
            The user-provided style for the feature.
        feature_spec : dict[str, Any]
            A dictionary containing the feature object and default styles.
        """
        if not style_arg:  # Allows for `coastlines=False`
            return

        style_kwargs = self._get_style(style_arg, feature_spec["defaults"])
        feature = feature_spec["feature"]
        self.ax.add_feature(feature, **style_kwargs)

    def add_features(self, **kwargs: Any) -> dict[str, Any]:
        """Add and style cartopy features on the map axes.

        This method provides a flexible, data-driven interface to add common
        map features. Features can be enabled with a boolean flag (e.g.,
        `coastlines=True`) or styled with a dictionary of keyword arguments
        (e.g., `states=dict(linewidth=2, edgecolor='red')`).

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
        # Note: The order of these calls is important.
        # Extent must be set before gridlines are drawn to ensure labels
        # are placed correctly.
        if "extent" in kwargs:
            extent = kwargs.pop("extent")
            self._set_extent(extent)

        if "gridlines" in kwargs:
            gridline_style = kwargs.pop("gridlines")
            self._draw_gridlines(gridline_style)

        # The rest of the kwargs are assumed to be for vector features.
        remaining_kwargs = self._draw_features(**kwargs)

        return remaining_kwargs

    def _set_extent(self, extent: tuple[float, float, float, float] | None) -> None:
        """Set the geographic extent of the map.

        Parameters
        ----------
        extent : tuple[float, float, float, float] | None
            The extent of the map as a tuple of (x_min, x_max, y_min, y_max).
            If None, the extent is not changed.
        """
        if extent is not None:
            self.ax.set_extent(extent)

    def _draw_features(self, **kwargs: Any) -> dict[str, Any]:
        """Draw vector features on the map.

        This is the primary feature-drawing loop, responsible for adding
        elements like coastlines, states, and borders.

        Parameters
        ----------
        **kwargs : Any
            Keyword arguments controlling the features to add and their
            styles.

        Returns
        -------
        dict[str, Any]
            A dictionary of the keyword arguments that were not used for
            adding features.
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

        return kwargs

    def _draw_gridlines(self, style: bool | dict[str, Any]) -> None:
        """Draw gridlines on the map.

        Parameters
        ----------
        style : bool or dict[str, Any]
            The style for the gridlines. If True, use defaults. If a dict,
            use it as keyword arguments. If False, do nothing.
        """
        if not style:
            return

        gridline_defaults = {
            "draw_labels": True,
            "linestyle": "--",
            "color": "gray",
        }
        gridline_kwargs = self._get_style(style, gridline_defaults)
        self.ax.gridlines(**gridline_kwargs)

    def _get_extent_from_data(
        self,
        data: xr.DataArray | xr.Dataset,
        lon_coord: str = "lon",
        lat_coord: str = "lat",
        buffer: float = 0.0,
    ) -> list[float]:
        """Calculate geographic extent from xarray data using Dask if available.

        Parameters
        ----------
        data : xr.DataArray or xr.Dataset
            The input data to calculate the extent from.
        lon_coord : str, optional
            Name of the longitude coordinate, by default "lon".
        lat_coord : str, optional
            Name of the latitude coordinate, by default "lat".
        buffer : float, optional
            Buffer to add to the extent as a fraction of the range,
            by default 0.0.

        Returns
        -------
        list[float]
            The calculated extent as [lon_min, lon_max, lat_min, lat_max].
        """
        lon = data[lon_coord]
        lat = data[lat_coord]

        # Use dask.compute for efficient parallel calculation of min/max
        # if the data is chunked.
        try:
            import dask

            lon_min, lon_max, lat_min, lat_max = dask.compute(
                lon.min(), lon.max(), lat.min(), lat.max()
            )
        except (ImportError, AttributeError):
            lon_min, lon_max = lon.min(), lon.max()
            lat_min, lat_max = lat.min(), lat.max()

        # Ensure they are scalar values (handles both numpy and dask returns)
        lon_min, lon_max = float(lon_min), float(lon_max)
        lat_min, lat_max = float(lat_min), float(lat_max)

        if buffer > 0:
            lon_range = lon_max - lon_min
            lat_range = lat_max - lat_min
            lon_buf = lon_range * buffer if lon_range > 0 else 1.0
            lat_buf = lat_range * buffer if lat_range > 0 else 1.0
            lon_min -= lon_buf
            lon_max += lon_buf
            lat_min -= lat_buf
            lat_max += lat_buf

        return [lon_min, lon_max, lat_min, lat_max]


class SpatialTrack(SpatialPlot):
    """Plot a trajectory from an xarray.DataArray on a map.

    This class provides an xarray-native interface for visualizing paths,
    such as flight trajectories or pollutant tracks, where a variable
    (e.g., altitude, concentration) is plotted along the path.

    It inherits from :class:`SpatialPlot` to provide the underlying map canvas.

    Attributes
    ----------
    data : xr.DataArray
        The trajectory data being plotted.
    lon_coord : str
        The name of the longitude coordinate in the DataArray.
    lat_coord : str
        The name of the latitude coordinate in the DataArray.
    """

    def __init__(
        self,
        data: xr.DataArray,
        *,
        lon_coord: str = "lon",
        lat_coord: str = "lat",
        **kwargs: Any,
    ):
        """Initialize the SpatialTrack plot.

        This constructor validates the input data and sets up the map canvas
        by initializing the parent `SpatialPlot` and adding map features.

        Parameters
        ----------
        data : xr.DataArray
            The input trajectory data. Must be an xarray DataArray with
            coordinates for longitude and latitude.
        lon_coord : str, optional
            Name of the longitude coordinate in the DataArray, by default 'lon'.
        lat_coord : str, optional
            Name of the latitude coordinate in the DataArray, by default 'lat'.
        **kwargs : Any
            Keyword arguments passed to :class:`SpatialPlot`. These control
            the map projection, figure size, and cartopy features. For example:
            `projection=ccrs.LambertConformal()`, `figsize=(10, 8)`,
            `states=True`, `extent=[-125, -70, 25, 50]`.
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

        # Set data and update history for provenance
        self.data = data
        self.lon_coord = lon_coord
        self.lat_coord = lat_coord
        _update_history(self.data, "Plotted with monet-plots.SpatialTrack")

    def plot(self, **kwargs: Any) -> plt.Artist:
        """Plot the trajectory on the map.

        The track is rendered as a scatter plot, where each point is colored
        according to the `data` values.

        Parameters
        ----------
        **kwargs : Any
            Keyword arguments passed to `matplotlib.pyplot.scatter`.
            Common options include `cmap`, `s` (size), and `alpha`.
            A `transform` keyword (e.g., `transform=ccrs.PlateCarree()`)
            is highly recommended for geospatial accuracy.
            The `cmap` argument can be a string, a Colormap object, or a
            (colormap, norm) tuple from the scaling tools in `colorbars.py`.
            Map features (e.g., `coastlines=True`) can also be passed here.

        Returns
        -------
        plt.Artist
            The scatter plot artist created by `ax.scatter`.
        """
        from ..plot_utils import get_plot_kwargs

        # Automatically compute extent if not provided
        if "extent" not in kwargs:
            kwargs["extent"] = self._get_extent_from_data(
                self.data, self.lon_coord, self.lat_coord, buffer=0.1
            )

        # Add features and get remaining kwargs for scatter
        scatter_kwargs = self.add_features(**kwargs)

        scatter_kwargs.setdefault("transform", ccrs.PlateCarree())

        # For coordinates and values, we pass the xarray objects directly.
        # This allows Matplotlib to handle the conversion, maintaining
        # compatibility with existing tests that check for lazy objects.
        longitude = self.data[self.lon_coord]
        latitude = self.data[self.lat_coord]

        # Use get_plot_kwargs to handle (cmap, norm) tuples
        final_kwargs = get_plot_kwargs(c=self.data, **scatter_kwargs)

        sc = self.ax.scatter(longitude, latitude, **final_kwargs)
        return sc
