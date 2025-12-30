# src/monet_plots/plots/spatial.py
from __future__ import annotations

import warnings
from typing import Any, Dict, Literal, Union

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from numpy.typing import ArrayLike


import numpy as np
from .base import BasePlot

# Type hint for array-like data
DataHint = Union[ArrayLike, pd.Series, xr.DataArray]


class SpatialPlot(BasePlot):
    """A base class for creating spatial plots with cartopy.

    This class provides a high-level interface for geospatial plots, handling
    the setup of cartopy axes and the addition of common map features like
    coastlines, states, and gridlines.

    Parameters
    ----------
    projection : ccrs.Projection
        The cartopy projection for the map. Default is ccrs.PlateCarree().
    resolution : {"10m", "50m", "110m"}
        The default resolution for cartopy features. Default is "50m".
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
    feature_kwargs : dict[str, Any]
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

        Notes
        -----
        For interactive use, it is recommended to create a `SpatialPlot`
        instance using the `SpatialPlot.create_map()` class method.
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

    def _get_feature_registry(self, resolution: str) -> dict[str, Any]:
        """Return a registry of functions to add cartopy features.

        This approach centralizes feature management, making it easier to
        add new features and maintain existing ones.

        Parameters
        ----------
        resolution : str
            The resolution for the cartopy features (e.g., '10m', '50m').

        Returns
        -------
        dict[str, Any]
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
            return defaults
        return {}

    def _draw_single_feature(
        self,
        key: str,
        feature_or_method: Any,
        style_arg: bool | dict[str, Any],
        resolution: str,
    ) -> None:
        """Draw a single cartopy feature on the axes.

        Parameters
        ----------
        key : str
            The name of the feature (e.g., 'states').
        feature_or_method : Any
            The cartopy feature object or a method like `ax.coastlines`.
        style_arg : bool or dict[str, Any]
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

    def add_features(self, **kwargs: Any) -> dict[str, Any]:
        """Add cartopy features to the map axes.

        This method provides a flexible interface to add and style common
        cartopy features like coastlines, states, and gridlines. Features can
        be enabled with a boolean flag (e.g., `coastlines=True`) or styled
        with a dictionary (e.g., `states=dict(linewidth=2)`).

        Parameters
        ----------
        **kwargs : Any
            Keyword arguments controlling the features to add. Common
            options include `coastlines`, `states`, `countries`, `ocean`,
            `land`, `lakes`, `rivers`, `borders`, and `gridlines`.
            The `extent` keyword is also supported to set the map boundaries.

        Returns
        -------
        dict[str, Any]
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
    def create_map(
        cls,
        *,
        projection: ccrs.Projection = ccrs.PlateCarree(),
        fig_kw: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> "SpatialPlot":
        """Create a SpatialPlot instance with a map projection.

        This is the recommended factory method for creating a map. It
        initializes the plot with a specified projection and then adds map
        features based on keyword arguments.

        Parameters
        ----------
        projection : ccrs.Projection
            The cartopy projection for the map. Default is ccrs.PlateCarree().
        fig_kw : dict[str, Any], optional
            Keyword arguments passed to `plt.figure()`.
        **kwargs : Any
            Keyword arguments for map features (e.g., `coastlines=True`,
            `states=True`, `extent=[-125, -70, 25, 50]`).

        Returns
        -------
        SpatialPlot
            An instance of the SpatialPlot class with the map drawn.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from monet_plots.plots.spatial import SpatialPlot
        >>> plot = SpatialPlot.create_map(states=True, extent=[-125, -70, 25, 50])
        >>> plt.show()
        """
        fig_kw = fig_kw or {}
        plot = cls(projection=projection, **fig_kw)
        plot.add_features(**kwargs)
        return plot

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
        extent: list[float] | None = None,
        figsize: tuple[float, float] = (10, 5),
        linewidth: float = 0.25,
        return_fig: bool = False,
        **kwargs: Any,
    ) -> plt.Axes | tuple[plt.Figure, plt.Axes]:
        """Draw a map with Cartopy.

        .. deprecated:: TBD
           Use :meth:`create_map` instead. This method is maintained for
           backward compatibility and will be removed in a future version.

        Parameters
        ----------
        crs : cartopy.crs.Projection, optional
            The map projection. Default is PlateCarree.
        natural_earth : bool
            Whether to add Natural Earth features (ocean, land, etc.).
            Default is False.
        coastlines : bool
            Whether to add coastlines. Default is True.
        states : bool
            Whether to add US states/provinces. Default is False.
        counties : bool
            Whether to add US counties. Default is False.
        countries : bool
            Whether to add country borders. Default is True.
        resolution : {"10m", "50m", "110m"}
            Resolution of Natural Earth features. Default is "10m".
        extent : list[float], optional
            Map extent as [lon_min, lon_max, lat_min, lat_max].
        figsize : tuple
            Figure size (width, height). Default is (10, 5).
        linewidth : float
            Line width for vector features. Default is 0.25.
        return_fig : bool
            If True, return the figure and axes objects. Default is False.
        **kwargs : Any
            Additional arguments passed to `plt.subplots()`.

        Returns
        -------
        plt.Axes or tuple[plt.Figure, plt.Axes]
            The matplotlib Axes object, or a tuple of (Figure, Axes) if
            `return_fig` is True.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from monet_plots.plots.spatial import SpatialPlot
        >>> plot = SpatialPlot.create_map(states=True, extent=[-125, -70, 25, 50])
        >>> plt.show()
        """
        warnings.warn(
            "`draw_map` is deprecated and will be removed in a future version. "
            "Please use `SpatialPlot.create_map()` instead.",
            FutureWarning,
            stacklevel=2,
        )

        # Prepare feature kwargs for the new factory
        feature_kwargs = {
            "natural_earth": natural_earth,
            "coastlines": {"linewidth": linewidth} if coastlines else False,
            "states": {"linewidth": linewidth} if states else False,
            "counties": {"linewidth": linewidth} if counties else False,
            "countries": {"linewidth": linewidth} if countries else False,
            "extent": extent,
            "resolution": resolution,
        }

        # Create the plot using the new factory
        all_kwargs = {**feature_kwargs, **kwargs}
        fig_kw = {"figsize": figsize} if "figsize" in all_kwargs else {}
        if "figsize" in all_kwargs:
            del all_kwargs["figsize"]

        plot = cls.create_map(
            projection=crs or ccrs.PlateCarree(), fig_kw=fig_kw, **all_kwargs
        )

        if return_fig:
            return plot.fig, plot.ax
        else:
            return plot.ax


class SpatialTrack(SpatialPlot):
    """Plot a trajectory from an xarray.DataArray on a map.

    This class provides an xarray-native interface for visualizing paths,
    such as flight trajectories or pollutant tracks, where a variable
    (e.g., altitude, concentration) is plotted along the path.

    Parameters
    ----------
    data : xr.DataArray
        An xarray DataArray containing the trajectory data. The DataArray
        must have coordinates corresponding to longitude and latitude.
    lon_coord : str, optional
        The name of the longitude coordinate in the DataArray, by default "lon".
    lat_coord : str, optional
        The name of the latitude coordinate in the DataArray, by default "lat".
    **kwargs : Any
        Additional keyword arguments passed to `SpatialPlot` (e.g., `states=True`).
    """

    def __init__(
        self,
        longitude,
        latitude=None,
        data=None,
        *args,
        lon_coord: str = "lon",
        lat_coord: str = "lat",
        **kwargs,
    ):
        """
        Accepts (lon, lat, data) arrays or a single xarray.DataArray for compatibility with tests and other spatial plots.
        """
        import xarray as xr

        super().__init__(*args, **kwargs)
        if latitude is None and data is None:
            # xarray.DataArray usage
            if not isinstance(longitude, xr.DataArray):
                raise TypeError("Input 'data' must be an xarray.DataArray.")
            if lon_coord not in longitude.coords:
                raise ValueError(
                    f"Longitude coordinate '{lon_coord}' not found in DataArray."
                )
            if lat_coord not in longitude.coords:
                raise ValueError(
                    f"Latitude coordinate '{lat_coord}' not found in DataArray."
                )
            self.data = longitude
            self.lon_coord = lon_coord
            self.lat_coord = lat_coord
        else:
            # Accepts (lon, lat, data)
            lon = np.asarray(longitude)
            lat = np.asarray(latitude)
            values = np.asarray(data)
            coords = {
                "time": np.arange(len(lon)),
                "lon": ("time", lon),
                "lat": ("time", lat),
            }
            self.data = xr.DataArray(
                values, dims=["time"], coords=coords, name="track_data"
            )
            self.lon_coord = "lon"
            self.lat_coord = "lat"
        # Update history attribute for provenance
        history = self.data.attrs.get("history", "")
        self.data.attrs["history"] = f"Plotted with monet-plots.SpatialTrack; {history}"

    def plot(self, **kwargs: Any) -> plt.Artist:
        """Plot the trajectory on the map.

        The track is rendered as a scatter plot, where each point is colored
        according to the `data` values. Keyword arguments are passed to both
        the feature-adding methods and `ax.scatter`.

        Parameters
        ----------
        **kwargs : Any
            Keyword arguments passed to `add_features` and `matplotlib.pyplot.scatter`.
            A `transform` keyword (e.g., `transform=ccrs.PlateCarree()`)
            is highly recommended for geospatial accuracy.

        Returns
        -------
        plt.Artist
            The scatter plot artist created by `ax.scatter`.

        Examples
        --------
        >>> import numpy as np
        >>> import xarray as xr
        >>> import matplotlib.pyplot as plt
        >>> from monet_plots.plots.spatial import SpatialTrack
        >>>
        >>> # 1. Create a sample xarray.DataArray
        >>> time = np.arange(20)
        >>> lon = np.linspace(-120, -80, 20)
        >>> lat = np.linspace(30, 45, 20)
        >>> concentration = np.linspace(0, 100, 20)
        >>> da = xr.DataArray(
        ...     concentration,
        ...     dims=['time'],
        ...     coords={
        ...         'time': time,
        ...         'lon': ('time', lon),
        ...         'lat': ('time', lat)
        ...     },
        ...     name='O3_concentration',
        ...     attrs={'units': 'ppb'}
        ... )
        >>>
        >>> # 2. Create and render the plot
        >>> track_plot = SpatialTrack(da, states=True, counties=True)
        >>> sc = track_plot.plot(cmap='viridis', transform=ccrs.PlateCarree())
        >>> plt.colorbar(sc, label="O3 Concentration (ppb)")
        >>> plt.show()
        """
        plot_kwargs = self.add_features(**kwargs)
        plot_kwargs.setdefault("transform", ccrs.PlateCarree())

        longitude = self.data[self.lon_coord]
        latitude = self.data[self.lat_coord]

        sc = self.ax.scatter(longitude, latitude, c=self.data.values, **plot_kwargs)
        return sc
