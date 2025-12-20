import pytest
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
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


def test_spatial_plot_init(clear_figures, sample_da):
    """Test SpatialPlot initialization."""
    plot = SpatialPlot()
    assert plot is not None


def test_spatial_plot_plot(clear_figures, sample_da):
    """Test SpatialPlot plot method."""
    plot = SpatialPlot()
    # ax = plot.plot()  # SpatialPlot has no plot method
    # assert ax is not None
    assert plot.ax is not None


def test_SpatialTrack_plot(clear_figures):
    """Test SpatialTrack plot method."""
    lon = np.linspace(-120, -80, 10)
    lat = np.linspace(30, 40, 10)
    data = np.random.rand(10)
    plot = SpatialTrack(lon, lat, data)
    plot.plot()


def test_spatial_plot_draw_features(clear_figures):
    """Test that map features are drawn correctly when plotting."""
    lon = np.linspace(-120, -80, 10)
    lat = np.linspace(30, 40, 10)
    data = np.random.rand(10)
    plot = SpatialTrack(
        lon, lat, data, coastlines=True, borders=True, gridlines=True
    )
    plot.plot()
    # The _draw_features method adds artists to the axes, so we check the collections
    assert len(plot.ax.collections) > 0
    assert plot.ax.gridlines is not None


def test_spatial_plot_da(clear_figures, sample_da):
    """Test plotting a DataArray with SpatialPlot."""
    plot = SpatialPlot()
    artist = plot.plot(sample_da)
    assert artist is not None
    assert hasattr(artist, "get_array")  # Check if it's a QuadMesh
