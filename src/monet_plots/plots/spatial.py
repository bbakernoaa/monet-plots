# src/monet_plots/plots/spatial.py
from __future__ import annotations

from typing import Any, Dict, List, Literal, Tuple, Union

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from numpy.typing import ArrayLike

from .base import BasePlot

# Type hint for array-like data
DataHint = Union[ArrayLike, pd.Series, xr.DataArray]


class SpatialPlot(BasePlot):
    """Base class for creating spatial plots with cartopy.

    This class provides a high-level interface for geospatial plots, handling
    the setup of cartopy axes and the addition of common map features like
    coastlines, states, and gridlines.

    Parameters
    ----------
    projection : ccrs.Projection, optional
        The cartopy projection for the map, by default ccrs.PlateCarree().
    resolution : {"10m", "50m", "110m"}, optional
        The default resolution for cartopy features, by default "50m".
    fig : plt.Figure, optional
        An existing matplotlib Figure object. If None, a new one is created.
    ax : plt.Axes, optional
        An existing matplotlib Axes object. If None, a new one is created.
    **kwargs : Any
        Additional keyword arguments for `monet_plots.plots.base.BasePlot`
        (e.g., `figsize`) and cartopy features (e.g., `coastlines=True`).

    Attributes
    ----------
    resolution : str
        The resolution of the cartopy features (e.g., '50m').
    feature_kwargs : Dict[str, Any]
        Keyword arguments for cartopy features passed during initialization.
    """

    def __init__(
        self,
        *,
        projection: ccrs.Projection = ccrs.PlateCarree(),
        resolution: Literal["10m", "50m", "110m"] = "50m",
        fig: plt.Figure | None = None,
        ax: plt.Axes | None = None,
        **kwargs: Any,
    ):
        """Initialize the spatial plot."""
        # The 'projection' kwarg is passed to subplot creation via 'subplot_kw'.
        subplot_kw = kwargs.pop("subplot_kw", {})
        subplot_kw["projection"] = projection

        # Extract kwargs for BasePlot.
        # Assumes that any kwargs not used by plt.subplots are feature kwargs.
        base_plot_kwargs: Dict[str, Any] = {"subplot_kw": subplot_kw}
        if "figsize" in kwargs:
            base_plot_kwargs["figsize"] = kwargs.pop("figsize")

        # The remaining kwargs are for features
        self.feature_kwargs = kwargs
        self.resolution = resolution

        super().__init__(fig=fig, ax=ax, **base_plot_kwargs)

    def _get_feature_registry(self, resolution: str) -> Dict[str, Any]:
        """Return a registry of functions to add cartopy features.

        This approach centralizes feature management, making it easier to
        add new features and maintain existing ones.

        Parameters
        ----------
        resolution : str
            The resolution for the cartopy features (e.g., '10m', '50m').

        Returns
        -------
        Dict[str, Any]
            A dictionary mapping feature names to the corresponding cartopy
            feature objects or methods.
        """
        # Lazy import to avoid circular dependencies if this were ever needed.
        # For now, it's just a clean way to organize.
        from cartopy.feature import BORDERS, LAKES, LAND, OCEAN, RIVERS, STATES

        feature_mapping = {
            # Special handling for ax.coastlines
            "coastlines": self.ax.coastlines,
            # Natural Earth Features
            "countries": BORDERS.with_scale(resolution),
            "states": STATES.with_scale(resolution),
            "borders": BORDERS,
            "ocean": OCEAN,
            "land": LAND,
            "rivers": RIVERS,
            "lakes": LAKES,
            # Special case for counties
            "counties": cfeature.NaturalEarthFeature(
                category="cultural",
                name="admin_2_counties",
                scale=resolution,
                facecolor="none",
                edgecolor="k",
            ),
            # Special handling for ax.gridlines
            "gridlines": self.ax.gridlines,
        }
        return feature_mapping

    @staticmethod
    def _get_style(
        style: bool | Dict[str, Any], defaults: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        """Helper to derive a style dictionary for a feature.

        Parameters
        ----------
        style : bool or Dict[str, Any]
            The user-provided style. If True, use defaults. If a dict, use it.
        defaults : Dict[str, Any], optional
            The default style to apply if `style` is True, by default None.

        Returns
        -------
        Dict[str, Any]
            The resolved keyword arguments for styling.
        """
        if isinstance(style, dict):
            return style
        if style and defaults:
            return defaults
        return {}

    def _draw_single_feature(
        self,
        key: str,
        feature_or_method: Any,
        style_arg: bool | Dict[str, Any],
        resolution: str,
    ) -> None:
        """Draw a single cartopy feature on the axes.

        Parameters
        ----------
        key : str
            The name of the feature (e.g., 'states').
        feature_or_method : Any
            The cartopy feature object or a method like `ax.coastlines`.
        style_arg : bool or Dict[str, Any]
            The user-provided style for the feature.
        resolution : str
            The resolution to use for scalable features.
        """
        if not style_arg:  # Allows for `coastlines=False`
            return

        # Determine default styles based on the feature
        defaults = {}
        if key in ["coastlines", "counties", "states", "countries", "borders"]:
            defaults = {"linewidth": 0.5, "edgecolor": "black"}
        elif key == "gridlines":
            defaults = {"draw_labels": True, "linestyle": "--", "color": "gray"}

        style_kwargs = self._get_style(style_arg, defaults)

        # Draw the feature
        if callable(feature_or_method):  # ax.coastlines or ax.gridlines
            if key == "coastlines":
                style_kwargs["resolution"] = resolution
            feature_or_method(**style_kwargs)
        else:  # cfeature.Feature object
            self.ax.add_feature(feature_or_method, **style_kwargs)

    def _draw_features(self, **kwargs: Any) -> Dict[str, Any]:
        """Draw cartopy features on the map axes using a data-driven approach.

        This method iterates through a feature registry, drawing features that
        are requested either in the class constructor or in the plot call.

        Parameters
        ----------
        **kwargs : Any
            Keyword arguments for features to draw (e.g., `coastlines=True`).
            These take precedence over `__init__` kwargs.

        Returns
        -------
        Dict[str, Any]
            The remaining keyword arguments that were not used for features.
        """
        combined_kwargs = {**self.feature_kwargs, **kwargs}
        resolution = combined_kwargs.pop("resolution", self.resolution)
        feature_registry = self._get_feature_registry(resolution)

        # If natural_earth is True, enable a standard set of features
        if combined_kwargs.pop("natural_earth", False):
            for feature in ["ocean", "land", "lakes", "rivers"]:
                combined_kwargs.setdefault(feature, True)

        # Main feature-drawing loop
        for key, feature_or_method in feature_registry.items():
            if key in combined_kwargs:
                style_arg = combined_kwargs.pop(key)
                self._draw_single_feature(key, feature_or_method, style_arg, resolution)

        # Handle extent after features are drawn
        if "extent" in combined_kwargs:
            extent = combined_kwargs.pop("extent")
            if extent is not None:
                self.ax.set_extent(extent)

        return combined_kwargs

    @classmethod
    def draw_map(
        cls,
        *,
        crs: ccrs.Projection | None = None,
        natural_earth: bool = False,
        coastlines: bool = True,
        states: bool = False,
        counties: bool = False,
        countries: bool = True,
        resolution: Literal["10m", "50m", "110m"] = "10m",
        extent: List[float] | None = None,
        figsize: Tuple[float, float] = (10, 5),
        linewidth: float = 0.25,
        return_fig: bool = False,
        **kwargs: Any,
    ) -> plt.Axes | Tuple[plt.Figure, plt.Axes]:
        """Draw a map with Cartopy (legacy compatibility method).

        Creates a map with configurable features. This method is maintained for
        compatibility with the legacy `draw_map` function. The recommended
        approach is to instantiate `SpatialPlot` directly.

        Parameters
        ----------
        crs : cartopy.crs.Projection, optional
            The map projection, by default None (which uses PlateCarree).
        natural_earth : bool, optional
            Whether to add Natural Earth features (ocean, land, etc.),
            by default False.
        coastlines : bool, optional
            Whether to add coastlines, by default True.
        states : bool, optional
            Whether to add US states/provinces, by default False.
        counties : bool, optional
            Whether to add US counties, by default False.
        countries : bool, optional
            Whether to add country borders, by default True.
        resolution : {"10m", "50m", "110m"}, optional
            Resolution of Natural Earth features, by default "10m".
        extent : List[float], optional
            Map extent as [lon_min, lon_max, lat_min, lat_max], by default None.
        figsize : tuple, optional
            Figure size (width, height), by default (10, 5).
        linewidth : float, optional
            Line width for vector features, by default 0.25.
        return_fig : bool, optional
            If True, return the figure and axes objects, by default False.
        **kwargs : Any
            Additional arguments passed to `plt.subplots()`.

        Returns
        -------
        plt.Axes or Tuple[plt.Figure, plt.Axes]
            The matplotlib Axes object, or a tuple of (Figure, Axes) if
            `return_fig` is True.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from monet_plots.plots.spatial import SpatialPlot
        >>> ax = SpatialPlot.draw_map(states=True, extent=[-125, -70, 25, 50])
        >>> plt.show()
        """
        projection = crs or ccrs.PlateCarree()

        # Prepare feature kwargs
        feature_kwargs = {
            "natural_earth": natural_earth,
            "coastlines": {"linewidth": linewidth} if coastlines else False,
            "states": {"linewidth": linewidth} if states else False,
            "counties": {"linewidth": linewidth} if counties else False,
            "countries": {"linewidth": linewidth} if countries else False,
            "extent": extent,
        }

        # Create SpatialPlot instance
        all_kwargs = {**feature_kwargs, **kwargs}
        spatial_plot = cls(
            projection=projection, resolution=resolution, figsize=figsize, **all_kwargs
        )

        # Draw the features
        spatial_plot._draw_features()

        if return_fig:
            return spatial_plot.fig, spatial_plot.ax
        else:
            return spatial_plot.ax


class SpatialTrack(SpatialPlot):
    """Plot a trajectory on a map, with points colored by a variable.

    This class is useful for visualizing paths, such as flight trajectories
    or pollutant tracks, where a variable (e.g., altitude, concentration)
    is plotted along the path.

    Parameters
    ----------
    longitude : DataHint
        Longitude values for the track points. Can be a pandas Series,
        xarray DataArray, or numpy array.
    latitude : DataHint
        Latitude values for the track points.
    data : DataHint
        Data values used for coloring the track points.
    **kwargs : Any
        Additional keyword arguments passed to `SpatialPlot`.
    """

    def __init__(
        self,
        longitude: DataHint,
        latitude: DataHint,
        data: DataHint,
        **kwargs: Any,
    ):
        """Initialize the spatial track plot."""
        super().__init__(**kwargs)
        self.longitude = longitude
        self.latitude = latitude
        self.data = data

    def plot(self, **kwargs: Any) -> plt.Artist:
        """Plot the trajectory on the map.

        The track is rendered as a scatter plot, where each point is colored
        according to the `data` values provided during initialization.

        Parameters
        ----------
        **kwargs : Any
            Keyword arguments passed to `matplotlib.pyplot.scatter`.
            A `transform` keyword (e.g., `transform=ccrs.PlateCarree()`)
            is highly recommended.

        Returns
        -------
        plt.Artist
            The scatter plot artist created by `ax.scatter`.

        Examples
        --------
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from monet_plots.plots.spatial import SpatialTrack
        >>> lon = np.linspace(-120, -80, 20)
        >>> lat = np.linspace(30, 45, 20)
        >>> data = np.linspace(0, 100, 20)
        >>> track_plot = SpatialTrack(lon, lat, data, states=True)
        >>> sc = track_plot.plot(cmap='viridis')
        >>> plt.show()
        """
        plot_kwargs = self._draw_features(**kwargs)
        plot_kwargs.setdefault("transform", ccrs.PlateCarree())

        sc = self.ax.scatter(self.longitude, self.latitude, c=self.data, **plot_kwargs)
        return sc
