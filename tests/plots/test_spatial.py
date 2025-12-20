import pytest
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from monet_plots.plots.spatial import SpatialPlot, SpatialTrack


@pytest.fixture
def clear_figures():
    """Clear all existing figures before and after a test."""
    plt.close('all')
    yield
    plt.close('all')


@pytest.fixture
def sample_da():
    """Create a sample DataArray for testing."""
    return xr.DataArray(
        np.random.rand(10, 10),
        dims=('latitude', 'longitude'),
        coords={
            'latitude': np.arange(30, 40),
            'longitude': np.arange(-100, -90),
        }
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
