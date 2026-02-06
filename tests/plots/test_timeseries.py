import pytest
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from monet_plots.plots.timeseries import TimeSeriesPlot, TimeSeriesStatsPlot
import xarray as xr


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
            "time": pd.to_datetime(np.arange(10), unit="D"),
            "obs": np.random.rand(10),
            "variable": ["obs"] * 10,
            "units": ["-"] * 10,
        }
    )


def test_timeseries_plot_init(clear_figures, sample_data):
    """Test TimeSeriesPlot initialization."""
    plot = TimeSeriesPlot(df=sample_data, x="time", y="obs")
    assert plot is not None


def test_timeseries_plot_plot(clear_figures, sample_data):
    """Test TimeSeriesPlot plot method."""
    plot = TimeSeriesPlot(df=sample_data, x="time", y="obs")
    ax = plot.plot()
    assert ax is not None


def test_timeseries_stats_plot_init(clear_figures, sample_data):
    """Test TimeSeriesStatsPlot initialization."""
    plot = TimeSeriesStatsPlot(df=sample_data, col1="obs", col2="obs")
    assert plot is not None


def test_timeseries_stats_plot_plot_bias(clear_figures, sample_data):
    """Test TimeSeriesStatsPlot plot bias."""
    plot = TimeSeriesStatsPlot(df=sample_data, col1="obs", col2="obs")
    ax = plot.plot(stat="bias")
    assert ax is not None


def test_timeseries_stats_plot_xarray(clear_figures):
    """Test TimeSeriesStatsPlot with xarray input."""
    time = pd.date_range("2023-01-01", periods=10, freq="D")
    ds = xr.Dataset(
        {
            "obs": (["time"], np.random.rand(10)),
            "model": (["time"], np.random.rand(10)),
        },
        coords={"time": time},
    )
    plot = TimeSeriesStatsPlot(df=ds, col1="obs", col2="model")
    ax = plot.plot(stat="rmse", freq="W")
    assert ax is not None
