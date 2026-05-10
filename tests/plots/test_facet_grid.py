import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from monet_plots.plots.facet_grid import FacetGridPlot


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
            "x": np.random.rand(100),
            "y": np.random.rand(100),
            "row": np.random.choice(["a", "b"], 100),
            "col": np.random.choice(["c", "d"], 100),
        }
    )


def test_facet_grid_plot(clear_figures, sample_data):
    """Test FacetGridPlot with Pandas data."""
    plot = FacetGridPlot(data=sample_data, row="row", col="col")
    assert plot.grid is not None
    assert plot.fig is not None
    assert plot.ax is not None
    plot.close()


def test_facet_grid_xarray(clear_figures):
    """Test FacetGridPlot with Xarray data."""
    data = xr.DataArray(
        np.random.rand(2, 5, 5),
        coords={"time": [0, 1], "lat": np.arange(5), "lon": np.arange(5)},
        dims=["time", "lat", "lon"],
        name="test",
    )
    plot = FacetGridPlot(data, col="time")
    assert plot.is_xarray
    assert plot.grid is not None
    # Use robust axes access
    axes = getattr(plot.grid, "axs", None)
    if axes is None:
        axes = getattr(plot.grid, "axes", None)
    assert axes is not None
    plot.close()


def test_facet_grid_map_dataframe(clear_figures, sample_data):
    """Test map_dataframe method."""
    plot = FacetGridPlot(data=sample_data, col="col")
    plot.map_dataframe(plt.scatter, "x", "y")
    plot.close()


def test_facet_grid_set_titles(clear_figures, sample_data):
    """Test set_titles method."""
    plot = FacetGridPlot(data=sample_data, col="col")
    plot.set_titles("Col: {col_name}")
    plot.close()


def test_facet_grid_save(clear_figures, sample_data, tmp_path):
    """Test save method."""
    plot = FacetGridPlot(data=sample_data, col="col")
    out = tmp_path / "test_facet.png"
    plot.save(out)
    assert out.exists()
    plot.close()


def test_facet_grid_plot_method(clear_figures, sample_data):
    """Test plot method."""
    plot = FacetGridPlot(data=sample_data, col="col")
    plot.plot(plt.scatter, "x", "y")
    plot.close()
