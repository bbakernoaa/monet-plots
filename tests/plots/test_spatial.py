
import pytest
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from monet_plots.plots.spatial import SpatialPlot, SpatialTrack
import cartopy.crs as ccrs
from matplotlib.collections import PathCollection

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

def test_spatial_plot_instantiation(clear_figures):
    """Test that SpatialPlot can be instantiated with a specific projection."""
    projection = ccrs.LambertConformal()
    plot = SpatialPlot(projection=projection)
    assert plot is not None
    assert isinstance(plot.ax.projection, ccrs.LambertConformal)

def test_spatial_plot_draw_map(clear_figures):
    """Test the SpatialPlot.draw_map classmethod."""
    projection = ccrs.AlbersEqualArea()
    fig, ax = plt.subplots(subplot_kw={"projection": projection})
    initial_collection_count = len(ax.collections)
    ax = SpatialPlot.draw_map(crs=projection, states=True, ax=ax)
    assert ax is not None
    assert isinstance(ax.projection, ccrs.AlbersEqualArea)
    # Check if a new collection was added for the states feature
    assert len(ax.collections) > initial_collection_count


def test_SpatialTrack_plot(clear_figures):
    """Test SpatialTrack plot method with assertions."""
    lon = np.linspace(-120, -80, 10)
    lat = np.linspace(30, 40, 10)
    data = np.random.rand(10)
    plot = SpatialTrack(longitude=lon, latitude=lat, data=data)
    scatter = plot.plot()
    assert isinstance(scatter, PathCollection)
    # Check that the number of plotted points matches the input data
    assert len(scatter.get_offsets()) == len(lon)

def test_spatial_track_instantiation_no_args(clear_figures):
    """Test that SpatialTrack can be instantiated without positional arguments."""
    lon = np.linspace(-120, -80, 10)
    lat = np.linspace(30, 40, 10)
    data = np.random.rand(10)
    try:
        plot = SpatialTrack(longitude=lon, latitude=lat, data=data)
        assert plot is not None
    except TypeError:
        pytest.fail("SpatialTrack instantiation failed with a TypeError, indicating the *args fix was not successful.")
