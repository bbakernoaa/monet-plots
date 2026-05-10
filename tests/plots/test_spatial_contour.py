import datetime
from unittest.mock import MagicMock
import numpy as np
import pytest
import xarray as xr

try:
    import dask.array as da
except ImportError:
    da = None

from monet_plots.plots.spatial_contour import SpatialContourPlot


def create_mock_data(lazy=False):
    """Create mock xarray data for testing."""
    lon = np.linspace(-120, -70, 10)
    lat = np.linspace(25, 50, 10)
    data = np.random.rand(10, 10)

    da_xr = xr.DataArray(
        data, coords={"lon": lon, "lat": lat}, dims=("lat", "lon"), name="test_var"
    )

    if lazy and da is not None:
        da_xr = da_xr.chunk({"lat": 5, "lon": 5})

    return da_xr


def test_spatial_contour_plot():
    """Test the SpatialContourPlot plot method."""
    # Create a mock grid object
    mock_grid = MagicMock()
    mock_grid.variables = {
        "LAT": np.random.rand(1, 1, 10, 10),
        "LON": np.random.rand(1, 1, 10, 10),
    }

    # Create a sample model variable
    modelvar = np.random.rand(10, 10)

    # Create a SpatialContourPlot instance
    plot = SpatialContourPlot(modelvar, mock_grid, datetime.datetime.now(), ncolors=10)

    # Call the plot method
    c = plot.plot(cmap="viridis", levels=np.arange(0, 1.1, 0.1))

    # Assert that the plot objects are created
    assert c is not None
    plot.close()


def test_spatial_contour_plot_no_date():
    """Test the SpatialContourPlot plot method without a date."""
    # Create a mock grid object
    mock_grid = MagicMock()
    mock_grid.variables = {
        "LAT": np.random.rand(1, 1, 10, 10),
        "LON": np.random.rand(1, 1, 10, 10),
    }

    # Create a sample model variable
    modelvar = np.random.rand(10, 10)

    # Create a SpatialContourPlot instance
    plot = SpatialContourPlot(modelvar, mock_grid, ncolors=10)

    # Call the plot method
    c = plot.plot(cmap="viridis", levels=np.arange(0, 1.1, 0.1))

    # Assert that the plot objects are created
    assert c is not None
    plot.close()


def test_spatial_contour_plot_continuous():
    """Test the SpatialContourPlot plot method with a continuous colorbar."""
    # Create a mock grid object
    mock_grid = MagicMock()
    mock_grid.variables = {
        "LAT": np.random.rand(1, 1, 10, 10),
        "LON": np.random.rand(1, 1, 10, 10),
    }

    # Create a sample model variable
    modelvar = np.random.rand(10, 10)

    # Create a SpatialContourPlot instance
    plot = SpatialContourPlot(
        modelvar, mock_grid, datetime.datetime.now(), discrete=False, ncolors=10
    )

    # Call the plot method
    c = plot.plot(cmap="viridis", levels=np.arange(0, 1.1, 0.1))

    # Assert that the plot objects are created
    assert c is not None
    plot.close()


@pytest.mark.skipif(da is None, reason="dask not installed")
def test_contour_laziness():
    """Verify SpatialContourPlot handles lazy data without eager compute during init."""
    da_lazy = create_mock_data(lazy=True)

    # Initialize plot
    plot = SpatialContourPlot(da_lazy, None)

    # Verify data is still lazy
    assert hasattr(plot.modelvar.data, "chunks")
    # Verify history was updated
    assert (
        "Initialized monet-plots.SpatialContourPlot" in plot.modelvar.attrs["history"]
    )
    plot.close()


@pytest.mark.skipif(da is None, reason="dask not installed")
def test_contour_plot_parity():
    """Verify SpatialContourPlot produces an axes when plotted with both eager and lazy data."""
    eager_da = create_mock_data(lazy=False)
    lazy_da = create_mock_data(lazy=True)

    # Eager plot
    plot_eager = SpatialContourPlot(eager_da, None)
    ax_eager = plot_eager.plot(levels=5)
    assert ax_eager is not None
    plot_eager.close()

    # Lazy plot
    plot_lazy = SpatialContourPlot(lazy_da, None)
    ax_lazy = plot_lazy.plot(levels=5)
    assert ax_lazy is not None
    plot_lazy.close()
