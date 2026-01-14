# tests/plots/test_spatial.py
from unittest.mock import patch

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from cartopy.mpl.geoaxes import GeoAxes

from monet_plots.plots.spatial import SpatialPlot, SpatialTrack


@pytest.fixture
def spatial_plot():
    """A fixture for a default SpatialPlot instance."""
    return SpatialPlot(projection=ccrs.PlateCarree())


def test_spatial_plot_init_default():
    """Test SpatialPlot default initialization."""
    plot = SpatialPlot()
    assert isinstance(plot.fig, plt.Figure)
    assert isinstance(plot.ax, GeoAxes)
    assert isinstance(plot.ax.projection, ccrs.PlateCarree)


def test_spatial_plot_init_custom_fig_ax():
    """Test SpatialPlot initialization with existing figure and axes."""
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.LambertConformal()})
    plot = SpatialPlot(fig=fig, ax=ax)
    assert plot.fig is fig
    assert plot.ax is ax
    assert isinstance(plot.ax.projection, ccrs.LambertConformal)


def test_add_features_default_styles(spatial_plot):
    """Test adding features with default styles (e.g., states=True)."""
    # Force a render to ensure collections are populated before asserting
    spatial_plot.fig.canvas.draw()
    initial_collections = len(spatial_plot.ax.collections)

    # Add states and coastlines
    spatial_plot.add_features(states=True, coastlines=True)

    # Force a re-render to update the collections
    spatial_plot.fig.canvas.draw()

    # Assert that new collections were added to the axes
    # The exact number can be fragile, so we check it increased
    assert len(spatial_plot.ax.collections) > initial_collections


@patch("cartopy.mpl.geoaxes.GeoAxes.add_feature")
def test_add_features_custom_styles_mocked(mock_add_feature, spatial_plot):
    """Test that custom styles are passed to ax.add_feature."""
    custom_style = {"linewidth": 2.0, "edgecolor": "red"}
    spatial_plot.add_features(states=custom_style)

    # Assert that add_feature was called
    mock_add_feature.assert_called()

    # Assert that it was called with the correct custom styles
    args, kwargs = mock_add_feature.call_args
    assert kwargs.get("linewidth") == custom_style["linewidth"]
    assert kwargs.get("edgecolor") == custom_style["edgecolor"]


def test_add_features_disabled(spatial_plot):
    """Test that features are not added when the style is False."""
    spatial_plot.fig.canvas.draw()
    initial_collections = len(spatial_plot.ax.collections)

    spatial_plot.add_features(states=False, coastlines=False)
    spatial_plot.fig.canvas.draw()

    assert len(spatial_plot.ax.collections) == initial_collections


def test_from_projection_factory():
    """Test the from_projection classmethod factory."""
    # This tests if the factory correctly creates an instance and adds features
    plot = SpatialPlot.from_projection(
        projection=ccrs.AlbersEqualArea(),
        states=True,
        countries=True,
        extent=[-120, -70, 20, 50],
    )
    plot.fig.canvas.draw()

    assert isinstance(plot.ax, GeoAxes)
    assert isinstance(plot.ax.projection, ccrs.AlbersEqualArea)
    # Check that some features were added
    assert len(plot.ax.collections) > 0
    # Check that extent is set correctly
    # Note: Comparing extents is non-trivial for non-PlateCarree projections
    # as the values are in projected coordinates (meters), not degrees.
    # We assert that the extent has been changed from the default.
    assert plot.ax.get_extent() != (-180.0, 180.0, -90.0, 90.0)


def test_natural_earth_convenience_flag(spatial_plot):
    """Test the `natural_earth=True` convenience flag."""
    spatial_plot.fig.canvas.draw()
    initial_collections = len(spatial_plot.ax.collections)

    # The flag should add ocean, land, lakes, and rivers
    spatial_plot.add_features(natural_earth=True)
    spatial_plot.fig.canvas.draw()

    assert len(spatial_plot.ax.collections) > initial_collections


# --- SpatialTrack Tests ---


@pytest.fixture
def sample_trajectory_data():
    """Create a sample xarray.DataArray for trajectory plots."""
    time = np.arange(10)
    lon = np.linspace(-100, -90, 10)
    lat = np.linspace(30, 35, 10)
    data = np.linspace(0, 50, 10)
    da = xr.DataArray(
        data,
        dims=["time"],
        coords={"time": time, "lon": ("time", lon), "lat": ("time", lat)},
        name="O3",
        attrs={"units": "ppb"},
    )
    return da


def test_spatial_track_init(sample_trajectory_data):
    """Test SpatialTrack initialization."""
    track = SpatialTrack(sample_trajectory_data, projection=ccrs.PlateCarree())
    assert isinstance(track.data, xr.DataArray)
    assert "Plotted with monet-plots.SpatialTrack" in track.data.attrs["history"]


def test_spatial_track_missing_coords(sample_trajectory_data):
    """Test that SpatialTrack raises ValueError for missing coordinates."""
    with pytest.raises(ValueError, match="Longitude coordinate 'longitude' not found"):
        SpatialTrack(sample_trajectory_data, lon_coord="longitude")

    with pytest.raises(ValueError, match="Latitude coordinate 'latitude' not found"):
        SpatialTrack(sample_trajectory_data, lat_coord="latitude")


def test_spatial_track_plot_method(sample_trajectory_data):
    """Test the SpatialTrack plot method."""
    track = SpatialTrack(sample_trajectory_data, projection=ccrs.PlateCarree())
    scatter_artist = track.plot()

    assert isinstance(scatter_artist, plt.Artist)
    # Check if data is correctly passed to the scatter plot
    # Note: this is an indirect check
    assert len(scatter_artist.get_offsets()) == len(sample_trajectory_data["time"])
    np.testing.assert_array_equal(
        scatter_artist.get_array(), sample_trajectory_data.values
    )
