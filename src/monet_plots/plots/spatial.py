# src/monet_plots/plots/spatial.py

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from .base import BasePlot


class SpatialPlot(BasePlot):
    """Base class for spatial plots using cartopy.

    Handles the creation of cartopy axes and the drawing of common
    map features like coastlines, states, etc., via keyword arguments.
    """

    def __init__(self, projection=ccrs.PlateCarree(), *args, **kwargs):
        """
        Initialize the spatial plot.

        Args:
            projection (ccrs.Projection): The cartopy projection for the map.
            **kwargs: Additional keyword arguments for plotting, including cartopy features
                      like 'coastlines', 'countries', 'states', 'borders', 'ocean',
                      'land', 'rivers', 'lakes', 'gridlines'. These can be True for default
                      styling or a dict for custom styling.
        """
        # The 'projection' kwarg is passed to subplot creation via 'subplot_kw'.
        subplot_kw = kwargs.pop("subplot_kw", {})
        subplot_kw["projection"] = projection

        # Separate BasePlot kwargs from feature kwargs
        fig = kwargs.pop("fig", None)
        ax = kwargs.pop("ax", None)
        figsize = kwargs.pop("figsize", None)

        base_plot_kwargs = {}
        if fig:
            base_plot_kwargs["fig"] = fig
        if ax:
            base_plot_kwargs["ax"] = ax
        if figsize:
            base_plot_kwargs["figsize"] = figsize
        if subplot_kw:
            base_plot_kwargs["subplot_kw"] = subplot_kw

        super().__init__(*args, **base_plot_kwargs)

        # Store feature kwargs passed at initialization
        self.feature_kwargs = kwargs

    def _draw_features(self, **kwargs):
        """Draw cartopy features on the map axes based on kwargs."""
        # Combine kwargs from __init__ and the plot call, with plot() taking precedence
        combined_kwargs = {**self.feature_kwargs, **kwargs}

        # Pop off special kwargs that control features but aren't features themselves
        resolution = combined_kwargs.pop("resolution", "10m")
        linewidth = combined_kwargs.pop("linewidth", 0.25)
        extent = combined_kwargs.pop("extent", None)

        # Define a mapping from keyword to cartopy feature
        feature_map = {
            "coastlines": cfeature.COASTLINE,
            "countries": cfeature.BORDERS,
            "borders": cfeature.BORDERS,
            "ocean": cfeature.OCEAN,
            "land": cfeature.LAND,
            "rivers": cfeature.RIVERS,
            "lakes": cfeature.LAKES,
        }

        # Add Natural Earth features if requested
        if combined_kwargs.pop("natural_earth", False):
            self.ax.add_feature(cfeature.OCEAN)
            self.ax.add_feature(cfeature.LAND)
            self.ax.add_feature(cfeature.LAKES)
            self.ax.add_feature(cfeature.RIVERS)

        # Add states and counties with specified resolution
        if combined_kwargs.pop("states", False):
            states_provinces = cfeature.NaturalEarthFeature(
                category="cultural",
                name="admin_1_states_provinces_lines",
                scale=resolution,
                facecolor="none",
                edgecolor="k",
            )
            self.ax.add_feature(states_provinces, linewidth=linewidth)

        if combined_kwargs.pop("counties", False):
            counties = cfeature.NaturalEarthFeature(
                category="cultural",
                name="admin_2_counties",
                scale=resolution,
                facecolor="none",
                edgecolor="k",
            )
            self.ax.add_feature(counties, linewidth=linewidth)

        # Add other simple features
        for key, feature in feature_map.items():
            if key in combined_kwargs:
                style = combined_kwargs.pop(key)  # Remove from kwargs
                if isinstance(style, dict):
                    if key == "coastlines":
                        style.setdefault("resolution", resolution)
                        self.ax.coastlines(**style)
                    else:
                        self.ax.add_feature(feature, **style)
                elif style:  # If it's True, add with default style
                    if key == "coastlines":
                        self.ax.coastlines(resolution=resolution, linewidth=linewidth)
                    else:
                        self.ax.add_feature(feature, linewidth=linewidth)

        # Special handling for gridlines
        if "gridlines" in combined_kwargs:
            gl_style = combined_kwargs.pop("gridlines")
            if isinstance(gl_style, dict):
                self.ax.gridlines(**gl_style)
            else:
                self.ax.gridlines(draw_labels=True, linestyle="--", color="gray")

        # Set map extent
        if extent is not None:
            self.ax.set_extent(extent)

        # Return the remaining kwargs so they can be passed to the actual plotting function
        return combined_kwargs

    def plot(self, data=None, **kwargs):
        """Plot an xarray DataArray and draw map features."""
        plot_kwargs = self._draw_features(**kwargs)
        if data is not None:
            return data.plot(ax=self.ax, **plot_kwargs)
        return self.ax


class SpatialTrack(SpatialPlot):
    """Plot a trajectory on a map, with points colored by a variable."""

    def __init__(self, longitude, latitude, data, *args, **kwargs):
        """
        Initialize the spatial track plot.
        Args:
            longitude (np.ndarray, pd.Series, xr.DataArray): Longitude values.
            latitude (np.ndarray, pd.Series, xr.DataArray): Latitude values.
            data (np.ndarray, pd.Series, xr.DataArray): Data to use for coloring the track.
            **kwargs: Additional keyword arguments passed to SpatialPlot.
        """
        super().__init__(*args, **kwargs)
        self.longitude = longitude
        self.latitude = latitude
        self.data = data

    def plot(self, **kwargs):
        """
        Plot the trajectory.
        Args:
            **kwargs: Keyword arguments passed to `matplotlib.pyplot.scatter`.
        """
        if self.ax is None:
            self.ax = self.fig.add_subplot(projection=self.projection)

        plot_kwargs = self._draw_features(**kwargs)
        plot_kwargs.setdefault("transform", ccrs.PlateCarree())

        sc = self.ax.scatter(self.longitude, self.latitude, c=self.data, **plot_kwargs)
        return sc
