import numpy as np
import pytest
import xarray as xr
from unittest.mock import MagicMock
import matplotlib.pyplot as plt

try:
    import dask.array as da
except ImportError:
    da = None

from monet_plots.plots.spatial_imshow import SpatialImshowPlot


@pytest.fixture
def mock_grid():
    """Create a mock grid object."""
    grid = MagicMock()
    grid.variables = {
        "LAT": np.random.uniform(low=20, high=50, size=(1, 1, 10, 10)),
        "LON": np.random.uniform(low=-120, high=-70, size=(1, 1, 10, 10)),
    }
    return grid


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


def test_spatial_imshow_plot(mock_grid):
    """Test the SpatialImshowPlot plot method."""
    # Create a sample model variable
    modelvar = np.random.rand(10, 10)

    # Create a SpatialImshowPlot instance
    plot = SpatialImshowPlot(modelvar, mock_grid)

    # Call the plot method
    c = plot.plot()

    # Assert that the plot objects are created
    assert plot.fig is not None
    assert plot.ax is not None
    assert c is not None
    plot.close()


def test_spatial_imshow_plot_discrete(mock_grid):
    """Test the SpatialImshowPlot plot method with a discrete colorbar."""
    # Create a sample model variable
    modelvar = np.random.rand(10, 10)

    # Create a SpatialImshowPlot instance
    plot = SpatialImshowPlot(modelvar, mock_grid, discrete=True)

    # Call the plot method
    c = plot.plot()

    # Assert that the plot objects are created
    assert plot.fig is not None
    assert plot.ax is not None
    assert c is not None
    plot.close()


def test_spatial_imshow_plot_discrete_vmin_vmax(mock_grid):
    """Test the SpatialImshowPlot plot method with a discrete colorbar and vmin/vmax."""
    # Create a sample model variable
    modelvar = np.random.rand(10, 10)

    # Create a SpatialImshowPlot instance
    plot = SpatialImshowPlot(
        modelvar, mock_grid, discrete=True, plotargs={"vmin": 0.1, "vmax": 0.9}
    )

    # Call the plot method
    c = plot.plot()

    # Assert that the plot objects are created
    assert plot.fig is not None
    assert plot.ax is not None
    assert c is not None
    plot.close()


@pytest.mark.skipif(da is None, reason="dask not installed")
def test_spatial_imshow_eager_vs_lazy():
    """Verify SpatialImshowPlot handles both numpy and dask backends."""
    da_eager = create_mock_data(lazy=False)
    da_lazy = create_mock_data(lazy=True)

    # Track A: Eager
    plot_eager = SpatialImshowPlot(da_eager)
    assert isinstance(plot_eager.modelvar, xr.DataArray)

    cbar_eager = plot_eager.plot()
    assert cbar_eager is not None
    plot_eager.close()

    # Track B: Lazy
    plot_lazy = SpatialImshowPlot(da_lazy)
    assert hasattr(plot_lazy.modelvar.data, "dask")

    cbar_lazy = plot_lazy.plot()
    assert cbar_lazy is not None
    plot_lazy.close()


@pytest.mark.skipif(da is None, reason="dask not installed")
def test_imshow_laziness():
    """Verify SpatialImshowPlot handles lazy data without eager compute during init."""
    da_lazy = create_mock_data(lazy=True)

    # Initialize plot
    plot = SpatialImshowPlot(da_lazy)

    # Verify data is still lazy (has chunks)
    assert hasattr(plot.modelvar.data, "chunks")
    # Verify history was updated
    assert "Initialized monet-plots.SpatialImshowPlot" in plot.modelvar.attrs["history"]
    plot.close()


@pytest.mark.skipif(da is None, reason="dask not installed")
def test_imshow_plot_parity():
    """Verify SpatialImshowPlot produces an axes when plotted with both eager and lazy data."""
    eager_da = create_mock_data(lazy=False)
    lazy_da = create_mock_data(lazy=True)

    # Eager plot
    plot_eager = SpatialImshowPlot(eager_da)
    ax_eager = plot_eager.plot()
    assert ax_eager is not None
    plot_eager.close()

    # Lazy plot (triggers compute internally for imshow)
    plot_lazy = SpatialImshowPlot(lazy_da)
    ax_lazy = plot_lazy.plot()
    assert ax_lazy is not None
    plot_lazy.close()


@pytest.mark.skipif(da is None, reason="dask not installed")
def test_spatial_imshow_auto_facet():
    """Verify SpatialImshowPlot automatically redirects to FacetGrid for multiple facets."""
    from monet_plots.plots.facet_grid import SpatialFacetGridPlot

    lon = np.linspace(-120, -70, 10)
    lat = np.linspace(25, 50, 10)
    time = np.arange(2)
    da_facet = xr.DataArray(
        np.random.rand(2, 10, 10),
        coords={"time": time, "lon": lon, "lat": lat},
        dims=("time", "lat", "lon"),
        name="test_var",
    ).chunk({"time": 1})

    # This should return a SpatialFacetGridPlot instance due to __new__
    plot = SpatialImshowPlot(da_facet, col="time")

    assert isinstance(plot, SpatialFacetGridPlot)
    assert plot.col == "time"
    plt.close("all")


@pytest.mark.skipif(da is None, reason="dask not installed")
def test_spatial_imshow_col_wrap():
    """Verify col_wrap works and triggers redirection."""
    from monet_plots.plots.facet_grid import SpatialFacetGridPlot

    lon = np.linspace(-120, -70, 10)
    lat = np.linspace(25, 50, 10)
    time = np.arange(4)
    da_wrap = xr.DataArray(
        np.random.rand(4, 10, 10),
        coords={"time": time, "lon": lon, "lat": lat},
        dims=("time", "lat", "lon"),
        name="test_var",
    )

    # Trigger with col_wrap
    plot = SpatialImshowPlot(da_wrap, col="time", col_wrap=2)

    assert isinstance(plot, SpatialFacetGridPlot)
    assert plot.col == "time"
    assert plot.col_wrap == 2
    plt.close("all")
