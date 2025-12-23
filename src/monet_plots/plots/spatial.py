# src/monet_plots/plots/spatial.py
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import ArrayLike

from .base import BasePlot

# Type hint for array-like data
DataHint = Union[ArrayLike, pd.Series, xr.DataArray]


class SpatialPlot(BasePlot):
    """Base class for spatial plots using cartopy.
    This class provides a foundation for creating geospatial plots. It handles
    the automatic creation of `cartopy` axes and provides a simple interface
    for adding common map features like coastlines, states, and gridlines.
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
        """Initialize the spatial plot.
        Parameters
        ----------
        projection : ccrs.Projection, optional
            The cartopy projection for the map, by default ccrs.PlateCarree().
        resolution : {"10m", "50m", "110m"}, optional
            Default resolution for cartopy features, by default "50m".
        fig : plt.Figure, optional
            An existing matplotlib Figure object. If None, a new one is created.
        ax : plt.Axes, optional
            An existing matplotlib Axes object. If None, a new one is created.
        **kwargs : Any
            Additional keyword arguments. These are split into two groups:
            1. Arguments for `monet_plots.plots.base.BasePlot` (e.g., `figsize`).
            2. Cartopy feature arguments (e.g., `coastlines=True`, `states=True`).
               These are stored and used when drawing the map.
        """
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

    def _add_natural_earth_features(self) -> None:
        """Add standard Natural Earth features (ocean, land, lakes, rivers)."""
        self.ax.add_feature(cfeature.OCEAN)
        self.ax.add_feature(cfeature.LAND)
        self.ax.add_feature(cfeature.LAKES)
        self.ax.add_feature(cfeature.RIVERS)

    def _add_coastlines(
        self, coastlines_style: bool | Dict[str, Any], resolution: str
    ) -> None:
        """Add coastlines with a specified style and resolution.
        Parameters
        ----------
        coastlines_style : Union[bool, Dict[str, Any]]
            If True, adds default coastlines. If a dict, uses it as keyword
            arguments for `ax.coastlines`.
        resolution : str
            The resolution for the coastlines feature (e.g., '10m').
        """
        if isinstance(coastlines_style, dict):
            linewidth = coastlines_style.pop("linewidth", 0.5)
            self.ax.coastlines(
                resolution=resolution, linewidth=linewidth, **coastlines_style
            )
        elif coastlines_style:
            self.ax.coastlines(resolution=resolution, linewidth=0.5)

    def _add_counties(
        self, counties_style: bool | Dict[str, Any], resolution: str
    ) -> None:
        """Add US counties feature.
        Parameters
        ----------
        counties_style : Union[bool, Dict[str, Any]]
            If True, adds default counties. If a dict, uses it as keyword
            arguments for `ax.add_feature`.
        resolution : str
            The resolution for the counties feature (e.g., '10m').
        """
        if counties_style:
            counties_feature = cfeature.NaturalEarthFeature(
                category="cultural",
                name="admin_2_counties",
                scale=resolution,
                facecolor="none",
                edgecolor="k",
            )
            if isinstance(counties_style, dict):
                self.ax.add_feature(counties_feature, **counties_style)
            else:
                self.ax.add_feature(counties_feature, linewidth=0.5)

    def _add_standard_features(
        self, feature_map: Dict[str, cfeature.Feature], combined_kwargs: Dict[str, Any]
    ) -> None:
        """Add a set of standard cartopy features from a mapping.
        Parameters
        ----------
        feature_map : Dict[str, cfeature.Feature]
            A dictionary mapping a feature name (e.g., "states") to a
            `cartopy.feature.Feature` object.
        combined_kwargs : Dict[str, Any]
            A dictionary of keyword arguments where keys matching the
            feature_map are used to style the feature.
        """
        for key, feature in feature_map.items():
            if key in combined_kwargs:
                style = combined_kwargs.pop(key)
                if isinstance(style, dict):
                    self.ax.add_feature(feature, **style)
                elif style:
                    self.ax.add_feature(feature, edgecolor="black", linewidth=0.5)

    def _add_gridlines(self, gl_style: bool | Dict[str, Any]) -> None:
        """Add gridlines to the map.
        Parameters
        ----------
        gl_style : Union[bool, Dict[str, Any]]
            If True, adds default gridlines. If a dict, uses it as keyword
            arguments for `ax.gridlines`.
        """
        if isinstance(gl_style, dict):
            self.ax.gridlines(**gl_style)
        else:
            self.ax.gridlines(draw_labels=True, linestyle="--", color="gray")

    def _draw_features(self, **kwargs: Any) -> Dict[str, Any]:
        """Draw cartopy features on the map axes based on kwargs.
        This method combines keyword arguments from the class initialization
        and the current plot call, then draws the corresponding cartopy
        features on the map.
        Parameters
        ----------
        **kwargs : Any
            Keyword arguments for features to draw (e.g., `coastlines=True`).
            These take precedence over `__init__` kwargs.
        Returns
        -------
        Dict[str, Any]
            The remaining keyword arguments that were not used to draw features.
        """
        # Combine kwargs from __init__ and the plot call, with plot() taking precedence
        combined_kwargs = {**self.feature_kwargs, **kwargs}

        # Get resolution from kwargs or use instance default
        resolution = combined_kwargs.pop("resolution", self.resolution)

        # Handle natural earth features first
        if combined_kwargs.pop("natural_earth", False):
            self._add_natural_earth_features()

        # Handle coastlines
        if "coastlines" in combined_kwargs:
            self._add_coastlines(combined_kwargs.pop("coastlines"), resolution)

        # Handle counties
        if "counties" in combined_kwargs:
            self._add_counties(combined_kwargs.pop("counties"), resolution)

        # Define feature mapping
        feature_map = {
            "countries": cfeature.BORDERS.with_scale(resolution),
            "states": cfeature.STATES.with_scale(resolution),
            "borders": cfeature.BORDERS,
            "ocean": cfeature.OCEAN,
            "land": cfeature.LAND,
            "rivers": cfeature.RIVERS,
            "lakes": cfeature.LAKES,
        }

        # Add standard features
        self._add_standard_features(feature_map, combined_kwargs)

        # Handle gridlines
        if "gridlines" in combined_kwargs:
            self._add_gridlines(combined_kwargs.pop("gridlines"))

        # Handle extent
        if "extent" in combined_kwargs:
            extent = combined_kwargs.pop("extent")
            if extent is not None:
                self.ax.set_extent(extent)

        # Return remaining kwargs
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
        """Draw a map with Cartopy - compatibility method for draw_map function.
        Creates a map using Cartopy with configurable features like coastlines,
        borders, and natural earth elements. This method provides compatibility
        with the legacy draw_map function.
        Parameters
        ----------
        crs : cartopy.crs.Projection, optional
            The map projection. Defaults to PlateCarree.
        natural_earth : bool, optional
            Add the Cartopy Natural Earth ocean, land, lakes, and rivers features.
        coastlines : bool, optional
            Add coastlines (linewidth applied).
        states : bool, optional
            Add states/provinces (linewidth applied).
        counties : bool, optional
            Add US counties (linewidth applied).
        countries : bool, optional
            Add country borders (linewidth applied).
        resolution : {'10m', '50m', '110m'}, optional
            The resolution of the Natural Earth features.
        extent : array-like, optional
            Set the map extent with [lon_min, lon_max, lat_min, lat_max].
        figsize : tuple, optional
            Figure size (width, height).
        linewidth : float, optional
            Line width for coastlines, states, counties, and countries.
        return_fig : bool, optional
            Return the figure and axes objects.
        **kwargs : Any
            Additional arguments passed to plt.subplots().
        Returns
        -------
        matplotlib.axes.Axes or tuple
            By default, returns just the axes. If return_fig is True,
            returns a tuple of (fig, ax).
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
    """Plot a trajectory on a map, with points colored by a variable."""

    def __init__(
        self,
        longitude: DataHint,
        latitude: DataHint,
        data: DataHint,
        **kwargs: Any,
    ):
        """Initialize the spatial track plot.
        Parameters
        ----------
        longitude : array-like
            Longitude values for the track points.
        latitude : array-like
            Latitude values for the track points.
        data : array-like
            Data values used for coloring the track points.
        **kwargs : Any
            Additional keyword arguments passed to `SpatialPlot`.
        """
        super().__init__(**kwargs)
        self.longitude = longitude
        self.latitude = latitude
        self.data = data

    def plot(self, **kwargs: Any) -> plt.Artist:
        """Plot the trajectory on the map.
        The track is rendered as a scatter plot, where each point is colored
        according to the data values.
        Parameters
        ----------
        **kwargs : Any
            Keyword arguments passed to `matplotlib.pyplot.scatter`.
        Returns
        -------
        matplotlib.artist.Artist
            The scatter plot artist.
        """
        plot_kwargs = self._draw_features(**kwargs)
        plot_kwargs.setdefault("transform", ccrs.PlateCarree())

        sc = self.ax.scatter(
            self.longitude, self.latitude, c=self.data, **plot_kwargs
        )
        return sc
