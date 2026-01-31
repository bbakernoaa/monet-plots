import pytest
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from monet_plots.plots.timeseries import TimeSeriesStatsPlot


@pytest.fixture
def clear_figures():
    """Clear all existing figures before and after a test."""
    plt.close("all")
    yield
    plt.close("all")


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    dates = pd.date_range("2023-01-01", periods=100, freq="h")
    return pd.DataFrame(
        {
            "time": dates,
            "obs": np.random.rand(100),
            "model1": np.random.rand(100),
            "model2": np.random.rand(100),
        }
    ).set_index("time")


def test_timeseries_stats_init(clear_figures, sample_df):
    """Test TimeSeriesStatsPlot initialization."""
    plot = TimeSeriesStatsPlot(df=sample_df, col1="obs", col2=["model1", "model2"])
    assert plot is not None
    assert plot.col1 == "obs"
    assert plot.col2 == ["model1", "model2"]


def test_timeseries_stats_plot_bias(clear_figures, sample_df):
    """Test TimeSeriesStatsPlot plot method with bias."""
    plot = TimeSeriesStatsPlot(df=sample_df, col1="obs", col2="model1")
    ax = plot.plot(stat="bias", freq="D")
    assert ax is not None
    assert ax.get_ylabel() == "BIAS"


def test_timeseries_stats_plot_rmse(clear_figures, sample_df):
    """Test TimeSeriesStatsPlot plot method with rmse."""
    plot = TimeSeriesStatsPlot(df=sample_df, col1="obs", col2="model1")
    ax = plot.plot(stat="rmse", freq="D")
    assert ax is not None
    assert ax.get_ylabel() == "RMSE"


def test_timeseries_stats_plot_corr(clear_figures, sample_df):
    """Test TimeSeriesStatsPlot plot method with corr."""
    plot = TimeSeriesStatsPlot(df=sample_df, col1="obs", col2="model1")
    ax = plot.plot(stat="corr", freq="D")
    assert ax is not None
    assert ax.get_ylabel() == "CORR"


def test_timeseries_stats_xarray(clear_figures, sample_df):
    """Test TimeSeriesStatsPlot with xarray input."""
    ds = sample_df.to_xarray()
    plot = TimeSeriesStatsPlot(df=ds, col1="obs", col2="model1")
    # TimeSeriesStatsPlot.normalize_data converts xr.Dataset to pd.DataFrame
    # so we can check that it works.
    ax = plot.plot(stat="bias", freq="D")
    assert ax is not None
    assert "Plotted TimeSeriesStatsPlot (bias)" in plot.df.attrs.get("history", "")


def test_timeseries_stats_invalid_stat(clear_figures, sample_df):
    """Test TimeSeriesStatsPlot with invalid statistic."""
    plot = TimeSeriesStatsPlot(df=sample_df, col1="obs", col2="model1")
    with pytest.raises(ValueError, match="not supported"):
        plot.plot(stat="invalid")
