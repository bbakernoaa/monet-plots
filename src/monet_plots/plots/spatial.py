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

        # Initialize the base plot, which creates the figure and axes.
        super().__init__(fig=fig, ax=ax, figsize=figsize, subplot_kw=current_subplot_kw)

        # Add features from kwargs
        # All remaining kwargs are passed to add_features to style the map.
        self.add_features(**kwargs)

    def _set_extent(self, extent: tuple[float, float, float, float] | None) -> None:
        """Set the geographical extent of the map.

        Parameters
        ----------
        extent : tuple[float, float, float, float] | None
            The extent of the map in the format (x0, x1, y0, y1).
        """
        if extent is not None:
            self.ax.set_extent(extent)

    def _draw_gridlines(self, style: bool | dict[str, Any]) -> None:
        """Draw gridlines on the map.

        Parameters
        ----------
        style : bool or dict
            The style for the gridlines. If True, use defaults.
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

        Special keywords `extent` and `gridlines` are also supported to
        set the map boundaries and draw gridlines, respectively.

        Parameters
        ----------
        **kwargs : Any
            Keyword arguments controlling the features to add and their
            styles. See `_get_feature_registry` for a list of supported
            cartopy features.

        Returns
        -------
        dict[str, Any]
            A dictionary of the keyword arguments that were not used for
            adding features. This can be useful for passing remaining
            arguments to other functions.
        """
        # Pop special (non-feature) arguments first.
        resolution = kwargs.pop("resolution", self.resolution)
        self._set_extent(kwargs.pop("extent", None))
        self._draw_gridlines(kwargs.pop("gridlines", False))

        # Get the feature registry
        feature_registry = self._get_feature_registry(resolution)

        # Handle the natural_earth convenience flag
        if kwargs.pop("natural_earth", False):
            for feature in ["ocean", "land", "lakes", "rivers"]:
                kwargs.setdefault(feature, True)

        # Main feature-drawing loop
        # Iterate over a copy since the dictionary is modified by pop
        remaining_kwargs = kwargs.copy()
        for key, style_arg in kwargs.items():
            if key in feature_registry:
                feature_spec = feature_registry[key]
                self._draw_single_feature(style_arg, feature_spec)
                remaining_kwargs.pop(key)

        return remaining_kwargs


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
        history = self.data.attrs.get("history", "")
        self.data.attrs["history"] = f"Plotted with monet-plots.SpatialTrack; {history}"

    def plot(self, **kwargs: Any) -> plt.Artist:
        """Plot the trajectory on the map.

        The track is rendered as a scatter plot, where each point is colored
        according to the `data` values.

        Parameters
        ----------
        **kwargs : Any
            Keyword arguments passed to `matplotlib.pyplot.scatter`.
            A `transform` keyword (e.g., `transform=ccrs.PlateCarree()`)
            is highly recommended for geospatial accuracy.
            The `cmap` argument can be a string, a Colormap object, or a
            (colormap, norm) tuple from the scaling tools in `colorbars.py`.

        Returns
        -------
        plt.Artist
            The scatter plot artist created by `ax.scatter`.
        """
        from ..plot_utils import get_plot_kwargs

        kwargs.setdefault("transform", ccrs.PlateCarree())

        longitude = self.data[self.lon_coord]
        latitude = self.data[self.lat_coord]

        # Use get_plot_kwargs to handle (cmap, norm) tuples
        final_kwargs = get_plot_kwargs(c=self.data, **kwargs)

        sc = self.ax.scatter(longitude, latitude, **final_kwargs)
        return sc
