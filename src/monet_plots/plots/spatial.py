# src/monet_plots/plots/spatial.py
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from .base import BasePlot

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    import xarray as xr
    from cartopy.feature import NaturalEarthFeature
    from matplotlib.axes import Axes
    from matplotlib.collections import PathCollection
    from matplotlib.figure import Figure


class SpatialPlot(BasePlot):
    """Base class for spatial plots using cartopy.

    Handles the creation of cartopy axes and the drawing of common
    map features like coastlines, states, etc., via keyword arguments.
    """

    def __init__(
        self,
        *,
        projection: ccrs.Projection = ccrs.PlateCarree(),
        resolution: str = "50m",
        **kwargs: Any,
    ) -> None:
        """Initialize the spatial plot.

        Parameters
        ----------
        projection : ccrs.Projection, optional
            The cartopy projection for the map, by default ccrs.PlateCarree().
        resolution : str, optional
            Resolution for cartopy features ('10m', '50m', '110m'), by default "50m".
        **kwargs : Any
            Additional keyword arguments for plotting, including cartopy features
            like 'coastlines', 'countries', 'states', 'borders', 'counties', 'ocean',
            'land', 'rivers', 'lakes', 'gridlines', 'natural_earth'. These can be True for default
            styling or a dict for custom styling.
        """
        # The 'projection' kwarg is passed to subplot creation via 'subplot_kw'.
        subplot_kw = kwargs.pop("subplot_kw", {})
        subplot_kw["projection"] = projection

        # Separate BasePlot kwargs from feature kwargs
        base_plot_kwargs: Dict[str, Any] = {"subplot_kw": subplot_kw}
        for key in ["fig", "ax", "figsize"]:
            if key in kwargs:
                base_plot_kwargs[key] = kwargs.pop(key)

        super().__init__(**base_plot_kwargs)

        # Store resolution and feature kwargs passed at initialization
        self.resolution: str = resolution
        self.feature_kwargs: Dict[str, Any] = kwargs
        self.projection: ccrs.Projection = projection

    def _add_natural_earth_features(self) -> None:
        """Add natural earth features (ocean, land, lakes, rivers)."""
        self.ax.add_feature(cfeature.OCEAN)
        self.ax.add_feature(cfeature.LAND)
        self.ax.add_feature(cfeature.LAKES)
        self.ax.add_feature(cfeature.RIVERS)

    def _add_coastlines(self, coastlines_style: Union[bool, Dict[str, Any]], resolution: str) -> None:
        """Add coastlines with proper resolution."""
        if isinstance(coastlines_style, dict):
            linewidth = coastlines_style.pop("linewidth", 0.5)
            self.ax.coastlines(resolution=resolution, linewidth=linewidth, **coastlines_style)
        elif coastlines_style:
            self.ax.coastlines(resolution=resolution, linewidth=0.5)

    def _add_counties(self, counties_style: Union[bool, Dict[str, Any]], resolution: str) -> None:
        """Add US counties feature."""
        if counties_style:
            counties_feature: NaturalEarthFeature = cfeature.NaturalEarthFeature(
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
        self, feature_map: Dict[str, NaturalEarthFeature], combined_kwargs: Dict[str, Any]
    ) -> None:
        """Add standard cartopy features."""
        for key, feature in feature_map.items():
            if key in combined_kwargs:
                style = combined_kwargs.pop(key)
                if isinstance(style, dict):
                    self.ax.add_feature(feature, **style)
                elif style:
                    self.ax.add_feature(feature, edgecolor="black", linewidth=0.5)

    def _add_gridlines(self, gl_style: Union[bool, Dict[str, Any]]) -> None:
        """Add gridlines to the map."""
        if isinstance(gl_style, dict):
            self.ax.gridlines(**gl_style)
        else:
            self.ax.gridlines(draw_labels=True, linestyle="--", color="gray")

    def _draw_features(self, **kwargs: Any) -> Dict[str, Any]:
        """Draw cartopy features on the map axes based on kwargs."""
        # Combine kwargs from __init__ and the plot call, with plot() taking precedence
        combined_kwargs = {**self.feature_kwargs, **kwargs}

        # Get resolution from kwargs or use instance default
        resolution: str = combined_kwargs.pop("resolution", self.resolution)

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
        feature_map: Dict[str, NaturalEarthFeature] = {
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
            extent: Optional[List[float]] = combined_kwargs.pop("extent")
            if extent is not None:
                self.ax.set_extent(extent)

        # Return remaining kwargs
        return combined_kwargs

    @classmethod
    def draw_map(
        cls,
        *,
        crs: Optional[ccrs.Projection] = None,
        natural_earth: bool = False,
        coastlines: bool = True,
        states: bool = False,
        counties: bool = False,
        countries: bool = True,
        resolution: str = "10m",
        extent: Optional[List[float]] = None,
        figsize: Tuple[float, float] = (10, 5),
        linewidth: float = 0.25,
        return_fig: bool = False,
        **kwargs: Any,
    ) -> Union[Axes, Tuple[Figure, Axes]]:
        """Draw a map with Cartopy - compatibility method for draw_map function.

        Creates a map using Cartopy with configurable features like coastlines,
        borders, and natural earth elements. This method provides compatibility
        with the legacy draw_map function.

        Parameters
        ----------
        crs : Optional[ccrs.Projection], optional
            The map projection, by default None which defaults to PlateCarree.
        natural_earth : bool, optional
            Add the Cartopy Natural Earth ocean, land, lakes, and rivers features, by default False.
        coastlines : bool, optional
            Add coastlines (linewidth applied), by default True.
        states : bool, optional
            Add states/provinces (linewidth applied), by default False.
        counties : bool, optional
            Add US counties (linewidth applied), by default False.
        countries : bool, optional
            Add country borders (linewidth applied), by default True.
        resolution : str, optional
            The resolution of the Natural Earth features, by default "10m".
        extent : Optional[List[float]], optional
            Set the map extent with [lon_min, lon_max, lat_min, lat_max], by default None.
        figsize : Tuple[float, float], optional
            Figure size (width, height), by default (10, 5).
        linewidth : float, optional
            Line width for coastlines, states, counties, and countries, by default 0.25.
        return_fig : bool, optional
            Return the figure and axes objects, by default False.
        **kwargs : Any
            Additional arguments passed to `plt.subplots()`.

        Returns
        -------
        Union[Axes, Tuple[Figure, Axes]]
            By default, returns just the axes. If `return_fig` is True,
            returns a tuple of (fig, ax).
        """
        projection: ccrs.Projection = crs or ccrs.PlateCarree()

        # Prepare feature kwargs
        feature_kwargs: Dict[str, Any] = {
            "natural_earth": natural_earth,
            "coastlines": {"linewidth": linewidth} if coastlines else False,
            "states": {"linewidth": linewidth} if states else False,
            "counties": {"linewidth": linewidth} if counties else False,
            "countries": {"linewidth": linewidth} if countries else False,
            "extent": extent,
        }

        # Create SpatialPlot instance
        spatial_plot = cls(
            projection=projection, resolution=resolution, figsize=figsize, **feature_kwargs, **kwargs
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
        longitude: xr.DataArray | pd.Series | np.ndarray,
        latitude: xr.DataArray | pd.Series | np.ndarray,
        data: xr.DataArray | pd.Series | np.ndarray,
        **kwargs: Any,
    ) -> None:
        """Initialize the spatial track plot.

        Parameters
        ----------
        longitude : xr.DataArray | pd.Series | np.ndarray
            Longitude values for the track points.
        latitude : xr.DataArray | pd.Series | np.ndarray
            Latitude values for the track points.
        data : xr.DataArray | pd.Series | np.ndarray
            Data to use for coloring the track.
        **kwargs : Any
            Arbitrary keyword arguments passed to the parent `SpatialPlot` class.
        """
        super().__init__(**kwargs)
        self.longitude: xr.DataArray | pd.Series | np.ndarray = longitude
        self.latitude: xr.DataArray | pd.Series | np.ndarray = latitude
        self.data: xr.DataArray | pd.Series | np.ndarray = data

    def plot(self, **kwargs: Any) -> PathCollection:
        """
        Plot the trajectory.
        Args:
            **kwargs: Keyword arguments passed to `matplotlib.pyplot.scatter`.
        """
        if self.ax is None:
            self.ax = self.fig.add_subplot(projection=self.projection)

        plot_kwargs = self._draw_features(**kwargs)
        plot_kwargs.setdefault("transform", ccrs.PlateCarree())

        sc: PathCollection = self.ax.scatter(
            self.longitude, self.latitude, c=self.data, **plot_kwargs
        )
        return sc
