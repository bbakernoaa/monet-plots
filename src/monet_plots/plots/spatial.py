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
            **kwargs: Keyword arguments for plotting, including cartopy
                      features like 'coastlines', 'countries', 'states',
                      'borders', 'ocean', 'land', 'rivers', 'lakes',
                      'gridlines'.
        """
        # The 'projection' kwarg is passed to subplot creation via 'subplot_kw'.
        subplot_kw = kwargs.pop('subplot_kw', {})
        subplot_kw['projection'] = projection

        # Separate BasePlot kwargs from feature kwargs
        fig = kwargs.pop('fig', None)
        ax = kwargs.pop('ax', None)
        figsize = kwargs.pop('figsize', None)

        base_plot_kwargs = {}
        if fig:
            base_plot_kwargs['fig'] = fig
        if ax:
            base_plot_kwargs['ax'] = ax
        if figsize:
            base_plot_kwargs['figsize'] = figsize
        if subplot_kw:
            base_plot_kwargs['subplot_kw'] = subplot_kw

        super().__init__(*args, **base_plot_kwargs)

        # Store feature kwargs passed at initialization
        self.feature_kwargs = kwargs

    def _draw_features(self, **kwargs):
        """Draw cartopy features on the map axes based on kwargs."""
        # Combine kwargs from __init__ and the plot call.
        combined_kwargs = {**self.feature_kwargs, **kwargs}

        # Define a mapping from keyword to cartopy feature
        feature_map = {
            'coastlines': cfeature.COASTLINE,
            'countries': cfeature.BORDERS.with_scale('50m'),
            'states': cfeature.STATES.with_scale('50m'),
            'borders': cfeature.BORDERS,
            'ocean': cfeature.OCEAN,
            'land': cfeature.LAND,
            'rivers': cfeature.RIVERS,
            'lakes': cfeature.LAKES,
        }

        for key, feature in feature_map.items():
            if key in combined_kwargs:
                style = combined_kwargs.pop(key)  # Remove from kwargs
                if isinstance(style, dict):
                    self.ax.add_feature(feature, **style)
                elif style:  # If it's True, add with default style
                    self.ax.add_feature(
                        feature, edgecolor='black', linewidth=0.5)

        # Special handling for gridlines
        if 'gridlines' in combined_kwargs:
            gl_style = combined_kwargs.pop('gridlines')
            if isinstance(gl_style, dict):
                self.ax.gridlines(**gl_style)
            else:
                self.ax.gridlines(draw_labels=True,
                                  linestyle='--', color='gray')

        # Return the remaining kwargs.
        return combined_kwargs


class SpatialTrack(SpatialPlot):
    """Plot a trajectory on a map, with points colored by a variable."""

    def __init__(self, longitude, latitude, data, *args, **kwargs):
        """
        Initialize the spatial track plot.
        Args:
            longitude (np.ndarray, pd.Series, xr.DataArray): Longitude values.
            latitude (np.ndarray, pd.Series, xr.DataArray): Latitude values.
            data (np.ndarray, pd.Series, xr.DataArray): Data for coloring.
            **kwargs: Keyword arguments passed to SpatialPlot.
        """
        super().__init__(*args, **kwargs)
        self.longitude = longitude
        self.latitude = latitude
        self.data = data

    def plot(self, **kwargs):
        """
        Plot the trajectory.
        Args:
            **kwargs: Keyword arguments for `matplotlib.pyplot.scatter`.
        """
        if self.ax is None:
            self.ax = self.fig.add_subplot(projection=self.projection)

        plot_kwargs = self._draw_features(**kwargs)
        plot_kwargs.setdefault('transform', ccrs.PlateCarree())

        sc = self.ax.scatter(self.longitude, self.latitude,
                             c=self.data, **plot_kwargs)
        return sc
