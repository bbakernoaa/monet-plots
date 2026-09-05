import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from monet_plots.plots import KDEPlot


def test_kde_univariate_pandas():
    """Test univariate KDE with pandas DataFrame."""
    df = pd.DataFrame({"val": np.random.randn(100)})
    plot = KDEPlot(df, x="val", title="Test Univariate")
    ax = plot.plot(fill=True)

    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == "Test Univariate"
    plt.close(plot.fig)


def test_kde_bivariate_pandas():
    """Test bivariate KDE with pandas DataFrame."""
    df = pd.DataFrame({"x": np.random.randn(100), "y": np.random.randn(100)})
    plot = KDEPlot(df, x="x", y="y", title="Test Bivariate")
    ax = plot.plot(fill=True)

    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == "Test Bivariate"
    plt.close(plot.fig)


def test_kde_univariate_xarray():
    """Test univariate KDE with xarray DataArray."""
    da = xr.DataArray(np.random.randn(100), dims="sample", name="Temperature")
    plot = KDEPlot(da, x="Temperature")
    ax = plot.plot()

    assert isinstance(ax, plt.Axes)
    assert "Temperature" in da.attrs.get("history", "")
    plt.close(plot.fig)


def test_kde_lazy_dask():
    """Test KDE with lazy Dask-backed xarray."""
    try:
        import dask.array as da
    except ImportError:
        pytest.skip("Dask not installed")

    data = da.random.normal(0, 1, size=(1000,), chunks=(500,))
    xr_da = xr.DataArray(data, dims="x", name="lazy_val")

    plot = KDEPlot(xr_da, x="lazy_val")
    # sns.kdeplot will trigger compute, but we ensure initialization works
    ax = plot.plot()

    assert isinstance(ax, plt.Axes)
    plt.close(plot.fig)


def test_kde_legacy_keyword():
    """Test KDEPlot with legacy 'df' keyword argument."""
    df = pd.DataFrame({"val": np.random.randn(50)})
    plot = KDEPlot(data=None, df=df, x="val")
    ax = plot.plot()

    assert isinstance(ax, plt.Axes)
    plt.close(plot.fig)
