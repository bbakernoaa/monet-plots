import numpy as np
import pandas as pd
import pytest
import xarray as xr
from monet_plots.plots.kde import KDEPlot


@pytest.fixture
def sample_data():
    """Create a sample dataset for testing."""
    np.random.seed(42)
    x = np.random.normal(size=100)
    y = np.random.normal(size=100)
    df = pd.DataFrame({"x": x, "y": y})
    ds = xr.Dataset.from_dataframe(df)
    return ds


def test_kde_plot_init(sample_data):
    """Test KDEPlot initialization."""
    plot = KDEPlot(data=sample_data, x="x", title="Test KDE")
    assert plot.x == "x"
    assert plot.title == "Test KDE"
    assert "Initialized KDEPlot" in plot.data.attrs["history"]


def test_kde_plot_lazy_parity(sample_data):
    """Test parity between eager and lazy (Dask) data."""
    # Eager
    eager_plot = KDEPlot(data=sample_data, x="x")
    eager_ax = eager_plot.plot()

    # Lazy
    lazy_ds = sample_data.chunk({"index": 50})
    lazy_plot = KDEPlot(data=lazy_ds, x="x")
    lazy_ax = lazy_plot.plot()

    # Basic verification that plotting succeeded
    assert eager_ax is not None
    assert lazy_ax is not None
    assert "Generated KDEPlot" in lazy_plot.data.attrs["history"]


def test_kde_plot_bivariate(sample_data):
    """Test bivariate KDE plot."""
    plot = KDEPlot(data=sample_data, x="x", y="y")
    ax = plot.plot()
    assert ax is not None


def test_kde_plot_hvplot(sample_data):
    """Test hvplot method (Track B)."""
    pytest.importorskip("hvplot")
    # Univariate
    plot = KDEPlot(data=sample_data, x="x")
    hv_obj = plot.hvplot()
    assert hv_obj is not None

    # Bivariate
    plot_biv = KDEPlot(data=sample_data, x="x", y="y")
    hv_obj_biv = plot_biv.hvplot()
    assert hv_obj_biv is not None


def test_kde_plot_missing_x(sample_data):
    """Test that missing 'x' raises ValueError."""
    with pytest.raises(ValueError, match="Parameter 'x' must be provided."):
        KDEPlot(data=sample_data)
