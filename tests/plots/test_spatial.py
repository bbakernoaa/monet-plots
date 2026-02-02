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
def clear_figures():
    """Clear all existing figures before and after a test."""
    plt.close("all")
    yield
    plt.close("all")


@pytest.fixture
def spatial_plot(clear_figures):
    """A fixture for a default SpatialPlot instance."""
    return SpatialPlot(projection=ccrs.PlateCarree())


@pytest.fixture
def sample_da():
    """Create a sample DataArray for testing."""
    return xr.DataArray(
        np.random.rand(10, 10),
        dims=("latitude", "longitude"),
        coords={
            "latitude": np.arange(30, 40),
            "longitude": np.arange(-100, -90),
        },
    )


@pytest.fixture
def sample_dataarray():
    """Create a sample xarray.DataArray for testing."""
    time = np.arange(20)
    lon = np.linspace(-120, -80, 20)
    lat = np.linspace(30, 45, 20)
    concentration = np.linspace(0, 100, 20)
    da = xr.DataArray(
        concentration,
        dims=["time"],
        coords={"time": time, "lon": ("time", lon), "lat": ("time", lat)},
        name="O3_concentration",
        attrs={"units": "ppb"},
    )
    return da


def test_spatial_plot_init_default(clear_figures):
    """Test SpatialPlot default initialization with features."""
    plot = SpatialPlot()
    plot.fig.canvas.draw()
    assert isinstance(plot.fig, plt.Figure)
    assert isinstance(plot.ax, GeoAxes)
    assert isinstance(plot.ax.projection, ccrs.PlateCarree)
    # Coastlines should be added by default
    assert len(plot.ax.collections) > 0


def test_spatial_plot_init_with_features(clear_figures):
    """Test that the constructor can draw features directly."""
    # Initialize the plot with some features
    plot = SpatialPlot(
        projection=ccrs.AlbersEqualArea(),
        states=True,
        countries=True,
        extent=[-120, -70, 20, 50],
        resolution="110m",
    )
    plot.fig.canvas.draw()

    assert isinstance(plot.ax, GeoAxes)
    assert isinstance(plot.ax.projection, ccrs.AlbersEqualArea)
    # Check that some features were added
    assert len(plot.ax.collections) > 0
    # Check that extent is set correctly
    assert plot.ax.get_extent() != (-180.0, 180.0, -90.0, 90.0)


def test_spatial_plot_init_custom_fig_ax(clear_figures):
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
    spatial_plot.add_features(states=True, coastlines=True, resolution="110m")

    # Force a re-render to update the collections
    spatial_plot.fig.canvas.draw()

    # Assert that new collections were added to the axes
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


def test_spatial_plot_feature_styling(clear_figures):
    """Test that feature styling kwargs are correctly applied."""
    custom_style = {"linewidth": 2.5, "edgecolor": "red"}
    plot = SpatialPlot(states=custom_style, resolution="110m")
    # Features are added in __init__
    found_match = False
    for collection in plot.ax.collections:
        if (
            collection.get_linewidth()[0] == custom_style["linewidth"]
            and collection.get_edgecolor()[0][0] == 1.0  # Red
        ):
            found_match = True
            break
    assert found_match, "Failed to find a feature with the specified custom style."


# --- SpatialTrack Tests ---


def test_spatial_track_init(sample_dataarray, clear_figures):
    """Test SpatialTrack initialization."""
    track = SpatialTrack(sample_dataarray, projection=ccrs.PlateCarree())
    assert isinstance(track.data, xr.DataArray)
    assert "Plotted with monet-plots.SpatialTrack" in track.data.attrs["history"]


def test_spatial_track_missing_coords(sample_dataarray, clear_figures):
    """Test that SpatialTrack raises ValueError for missing coordinates."""
    with pytest.raises(
        ValueError, match="Longitude coordinate 'longitude' not found in DataArray."
    ):
        SpatialTrack(sample_dataarray, lon_coord="longitude")

    with pytest.raises(
        ValueError, match="Latitude coordinate 'latitude' not found in DataArray."
    ):
        SpatialTrack(sample_dataarray, lat_coord="latitude")


def test_spatialtrack_init_invalid_data_type(clear_figures):
    """Test that SpatialTrack raises TypeError for invalid data types."""
    with pytest.raises(TypeError, match="Input 'data' must be an xarray.DataArray."):
        SpatialTrack(data=np.zeros(5))


def test_spatial_track_plot_method(sample_dataarray, clear_figures):
    """Test the SpatialTrack plot method."""
    track = SpatialTrack(sample_dataarray, projection=ccrs.PlateCarree())
    scatter_artist = track.plot()

    assert isinstance(scatter_artist, plt.Artist)
    # Check if data is correctly passed to the scatter plot
    assert len(scatter_artist.get_offsets()) == len(sample_dataarray["time"])
    np.testing.assert_array_equal(scatter_artist.get_array(), sample_dataarray.values)


def test_spatialtrack_history_attribute_updated(sample_dataarray, clear_figures):
    """Test that the history attribute is correctly updated."""
    # Test case 1: No pre-existing history
    da_no_history = sample_dataarray.copy()
    if "history" in da_no_history.attrs:
        del da_no_history.attrs["history"]
    track_plot = SpatialTrack(data=da_no_history)
    assert "history" in track_plot.data.attrs
    assert "Plotted with monet-plots.SpatialTrack" in track_plot.data.attrs["history"]

    # Test case 2: Pre-existing history
    da_with_history = sample_dataarray.copy()
    da_with_history.attrs["history"] = "Initial analysis step."
    track_plot_2 = SpatialTrack(data=da_with_history)
    assert "Initial analysis step." in track_plot_2.data.attrs["history"]
    assert "Plotted with monet-plots.SpatialTrack" in track_plot_2.data.attrs["history"]


def test_spatialtrack_plot_is_lazy_with_dask(clear_figures):
    """Test that SpatialTrack.plot preserves lazy evaluation with dask."""
    dask = pytest.importorskip("dask")
    import dask.array as da

    # 1. Create lazy data
    time = np.arange(20)
    lon = np.linspace(-120, -80, 20)
    lat = np.linspace(30, 45, 20)
    concentration = da.from_array(np.linspace(0, 100, 20), chunks=(10,))
    da_lazy = xr.DataArray(
        concentration,
        dims=["time"],
        coords={"time": time, "lon": ("time", lon), "lat": ("time", lat)},
        name="O3_concentration_lazy",
    )

    # 2. Instantiate and plot, spying on the scatter call
    track_plot = SpatialTrack(data=da_lazy, states=True, resolution="110m")

    with patch.object(track_plot.ax, "scatter") as mock_scatter:
        track_plot.plot()
        # 3. Validation
        mock_scatter.assert_called_once()
        args, kwargs = mock_scatter.call_args
        c_arg = kwargs.get("c")

        # Ensure 'c' is an xarray.DataArray wrapping a dask array
        assert isinstance(c_arg, xr.DataArray), "The 'c' argument is not a DataArray."
        assert isinstance(
            c_arg.data, dask.array.Array
        ), "The underlying data is not a dask array."
