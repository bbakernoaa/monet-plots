# tests/plots/test_spatial.py
import pytest
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from monet_plots.plots.spatial import SpatialPlot, SpatialTrack


@pytest.fixture
def clear_figures():
    """Clear all existing figures before and after a test."""
    plt.close("all")
    yield
    plt.close("all")


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


def test_spatial_plot_init(clear_figures, sample_da):
    """Test SpatialPlot initialization."""
    plot = SpatialPlot()
    assert plot is not None


def test_spatial_plot_plot(clear_figures, sample_da):
    """Test SpatialPlot plot method."""
    plot = SpatialPlot()
    assert plot.ax is not None


def test_spatial_plot_draw_features_data_driven(clear_figures):
    """Test the data-driven feature drawing mechanism."""
    plot = SpatialPlot(
        states=True,
        coastlines=True,
        countries=True,
        land=True,
        ocean=True,
        resolution="110m",
    )
    initial_collections = len(plot.ax.collections)
    plot.add_features()
    final_collections = len(plot.ax.collections)
    assert final_collections > initial_collections
    assert final_collections >= 5


def test_spatial_plot_feature_styling(clear_figures):
    """Test that feature styling kwargs are correctly applied."""
    custom_style = {"linewidth": 2, "edgecolor": "red"}
    plot = SpatialPlot(states=custom_style, resolution="110m")
    plot.add_features()
    found_match = False
    for collection in plot.ax.collections:
        if (
            collection.get_linewidth()[0] == custom_style["linewidth"]
            and collection.get_edgecolor()[0][0] == 1.0
        ):
            found_match = True
            break
    assert found_match, "Failed to find a feature with the specified custom style."


def test_spatial_plot_draw_map_docstring_example(clear_figures):
    """Test the example from the SpatialPlot.draw_map docstring."""
    from cartopy.mpl.geoaxes import GeoAxes

    ax = SpatialPlot.draw_map(states=True, extent=[-125, -70, 25, 50])
    assert isinstance(ax, GeoAxes)
    assert len(ax.collections) > 0


def test_spatialplot_create_map(clear_figures):
    """Test the SpatialPlot.create_map factory method."""
    plot = SpatialPlot.create_map(states=True, extent=[-125, -70, 25, 50])
    assert isinstance(plot, SpatialPlot)
    from cartopy.mpl.geoaxes import GeoAxes

    assert isinstance(plot.ax, GeoAxes)
    assert len(plot.ax.collections) > 0


def test_spatialtrack_init_success(sample_dataarray):
    """Test successful initialization of SpatialTrack."""
    track_plot = SpatialTrack(data=sample_dataarray, states=True)
    assert track_plot.data is sample_dataarray
    assert track_plot.lon_coord == "lon"
    assert track_plot.lat_coord == "lat"
    assert "history" in track_plot.data.attrs


def test_spatialtrack_init_invalid_data_type():
    """Test that SpatialTrack raises TypeError for invalid data types."""
    with pytest.raises(TypeError, match="Input 'data' must be an xarray.DataArray."):
        SpatialTrack(data=np.zeros(5))


def test_spatialtrack_init_missing_lon_coord(sample_dataarray):
    """Test that SpatialTrack raises ValueError for missing longitude coordinate."""
    data_missing_lon = sample_dataarray.drop_vars("lon")
    with pytest.raises(
        ValueError, match="Longitude coordinate 'lon' not found in DataArray."
    ):
        SpatialTrack(data=data_missing_lon)


def test_spatialtrack_init_missing_lat_coord(sample_dataarray):
    """Test that SpatialTrack raises ValueError for missing latitude coordinate."""
    data_missing_lat = sample_dataarray.drop_vars("lat")
    with pytest.raises(
        ValueError, match="Latitude coordinate 'lat' not found in DataArray."
    ):
        SpatialTrack(data=data_missing_lat)


def test_spatialtrack_plot_runs(sample_dataarray):
    """Test that the plot method runs without errors."""
    track_plot = SpatialTrack(data=sample_dataarray, states=True)
    try:
        sc = track_plot.plot(cmap="viridis")
        assert sc is not None
        plt.close(track_plot.fig)
    except Exception as e:
        pytest.fail(f"SpatialTrack.plot() raised an exception: {e}")


def test_spatialtrack_history_attribute_updated(sample_dataarray):
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
