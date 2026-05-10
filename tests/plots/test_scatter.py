import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import cartopy.crs as ccrs

try:
    import dask.array as da
except ImportError:
    da = None

from monet_plots.plots.scatter import ScatterPlot


@pytest.fixture
def clear_figures():
    """Clear all existing figures before and after a test."""
    plt.close("all")
    yield
    plt.close("all")


@pytest.fixture
def sample_data():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "x": np.linspace(0, 10, 100),
            "y": np.linspace(0, 10, 100) + np.random.rand(100),
        }
    )


def test_scatter_plot_creates_plot(clear_figures, sample_data):
    """Test that ScatterPlot creates a plot."""
    plot = ScatterPlot(data=sample_data, x="x", y="y")
    ax = plot.plot()
    assert ax is not None
    assert len(ax.lines) > 0  # Check for regression line
    assert len(ax.collections) > 0  # Check for scatter points


def test_scatter_plot_with_c_and_colorbar(clear_figures, sample_data):
    """Test ScatterPlot with colorization and colorbar."""
    sample_data["c"] = np.random.rand(100)
    plot = ScatterPlot(data=sample_data, x="x", y="y", c="c", colorbar=True)
    ax = plot.plot()
    assert ax is not None
    assert len(ax.figure.axes) > 1  # Check for colorbar axes


@pytest.mark.skipif(da is None, reason="dask not installed")
def test_scatter_plot_eager_vs_lazy():
    """Verify ScatterPlot handles both numpy and dask backends."""
    # Create test data
    x = np.linspace(0, 10, 100)
    y = 2 * x + 1 + np.random.randn(100)

    ds_eager = xr.Dataset(
        {"x": (["index"], x), "y": (["index"], y)}, coords={"index": np.arange(100)}
    )
    ds_lazy = ds_eager.chunk({"index": 50})

    # Track A: Eager
    plot_eager = ScatterPlot(ds_eager, x="x", y="y")
    assert isinstance(plot_eager.data, xr.Dataset)

    ax_eager = plot_eager.plot()
    assert ax_eager is not None
    assert len(ax_eager.collections) > 0  # Scatter points
    assert len(ax_eager.lines) > 0  # Regression line
    plt.close(ax_eager.figure)

    # Track B: Lazy
    plot_lazy = ScatterPlot(ds_lazy, x="x", y="y")
    assert hasattr(plot_lazy.data.x.data, "dask")

    ax_lazy = plot_lazy.plot()
    assert ax_lazy is not None
    assert len(ax_lazy.collections) > 0
    assert len(ax_lazy.lines) > 0
    plt.close(ax_lazy.figure)


@pytest.mark.skipif(da is None, reason="dask not installed")
def test_scatter_plot_geospatial():
    """Verify ScatterPlot works on GeoAxes with mandatory transform."""
    x = np.linspace(-120, -70, 10)
    y = np.linspace(25, 50, 10)
    ds = xr.Dataset({"x": (["index"], x), "y": (["index"], y)})

    # Initialize with projection to create GeoAxes
    plot = ScatterPlot(ds, x="x", y="y", subplot_kw={"projection": ccrs.PlateCarree()})
    assert hasattr(plot.ax, "projection")

    # This should now NOT fail and apply the transform
    ax = plot.plot()
    assert ax is not None
    plt.close(ax.figure)
