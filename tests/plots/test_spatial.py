# tests/plots/test_spatial.py
from unittest.mock import patch
import cartopy.crs as ccrs
import pytest
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from cartopy.mpl.geoaxes import GeoAxes
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


def test_add_features_docstring_example(clear_figures):
    """Test the example from the add_features docstring."""
    plot = SpatialPlot.from_projection(
        projection=ccrs.LambertConformal(),
        figsize=(10, 5),
        states=True,
        coastlines=True,
        countries=True,
        extent=[-125, -65, 25, 50],
        resolution="110m",
    )
    initial_collections = len(plot.ax.collections)

    # Style the states with a dictionary
    unused_kwargs = plot.add_features(states=dict(linewidth=1.5, edgecolor="blue"))

    assert "states" not in unused_kwargs
    assert len(plot.ax.collections) > initial_collections


def test_spatial_plot_from_projection(clear_figures):
    """Test the from_projection factory to ensure it creates a map with the
    correct projection and adds features as expected."""
    # 1. The Logic (Implementation)
    plot = SpatialPlot.from_projection(
        projection=ccrs.LambertConformal(), states=True, coastlines=True
    )
    # Force a draw to update collections
    plot.fig.canvas.draw()

    # 2. The Proof (Validation)
    assert isinstance(plot.ax.projection, ccrs.LambertConformal)
    # Check that both states and coastlines were added
    assert len(plot.ax.collections) >= 2


def test_spatialtrack_init_success(sample_dataarray):
    """Test successful initialization of SpatialTrack."""
    track_plot = SpatialTrack(data=sample_dataarray)
    assert track_plot.data is sample_dataarray
    assert track_plot.lon_coord == "lon"
    assert track_plot.lat_coord == "lat"
    assert "history" in track_plot.data.attrs
    # With the refactor, init should not draw features
    assert len(track_plot.ax.collections) == 0


def test_spatialtrack_plot_draws_features_and_data(sample_dataarray, clear_figures):
    """Test that the plot method now draws map features and data."""
    # 1. ARRANGE: Initialize the plot object without any rendering
    track_plot = SpatialTrack(
        data=sample_dataarray,
        projection=ccrs.LambertConformal(),
        resolution="110m",
    )
    assert len(track_plot.ax.collections) == 0  # Should be empty initially

    # 2. ACT: Call the plot method to render features and data
    sc = track_plot.plot(states=True, coastlines=True, cmap="viridis")
    track_plot.fig.canvas.draw()  # Force render

    # 3. ASSERT: Verify that both features and data are drawn
    assert sc is not None, "Scatter artist should be returned"
    assert len(track_plot.ax.collections) > 0, "Cartopy features should be added"
    # The scatter plot adds a PathCollection
    assert any(
        isinstance(c, plt.matplotlib.collections.PathCollection)
        for c in track_plot.ax.collections
    )


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
    track_plot = SpatialTrack(data=sample_dataarray)
    try:
        sc = track_plot.plot(states=True, cmap="viridis")
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


def test_spatialtrack_plot_is_lazy_with_dask(clear_figures):
    """Test that SpatialTrack.plot passes a dask array to matplotlib."""
    dask = pytest.importorskip("dask")
    import dask.array as da

    # 1. The Logic (Create lazy data)
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

    # 2. The Proof (Instantiate and plot, spying on the scatter call)
    track_plot = SpatialTrack(data=da_lazy, resolution="110m")

    with patch.object(track_plot.ax, "scatter") as mock_scatter:
        track_plot.plot(states=True)
        # 3. The Validation
        mock_scatter.assert_called_once()
        args, kwargs = mock_scatter.call_args
        c_arg = kwargs.get("c")

        # Ensure 'c' is an xarray.DataArray wrapping a dask array
        assert isinstance(c_arg, xr.DataArray), "The 'c' argument is not a DataArray."
        assert isinstance(
            c_arg.data, dask.array.Array
        ), "The underlying data is not a dask array."


def test_spatial_track_inheritance_and_provenance(sample_dataarray):
    """Test SpatialTrack correctly inherits from SpatialPlot and tracks provenance."""
    # 1. ARRANGE: Add pre-existing history for a robust test
    sample_dataarray.attrs["history"] = "Original data."

    # 2. ACT: Create the plot instance
    track_plot = SpatialTrack(
        data=sample_dataarray,
        projection=ccrs.LambertConformal(),
    )
    track_plot.plot(states=True)

    # 3. ASSERT: Validate initialization, inheritance, and provenance
    assert isinstance(track_plot, SpatialPlot), "Should be a SpatialPlot subclass"
    assert isinstance(track_plot.ax, GeoAxes), "Axes should be a GeoAxes instance"
    assert track_plot.data is sample_dataarray, "Data attribute should be set correctly"
    # Force draw to ensure collections are updated
    track_plot.fig.canvas.draw()
    assert len(track_plot.ax.collections) > 0, "Cartopy features should be added"

    # Validate provenance tracking
    history = track_plot.data.attrs["history"]
    assert "Plotted with monet-plots.SpatialTrack" in history
    assert "Original data." in history
