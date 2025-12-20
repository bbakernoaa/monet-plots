import pytest
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from monet_plots.plots.timeseries import TimeSeriesPlot


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


from monet_plots.plots.timeseries import TimeSeriesStatsPlot


def test_timeseries_plot_init(clear_figures, sample_data):
    """Test TimeSeriesPlot initialization."""
    plot = TimeSeriesPlot(data=sample_data, x="time", y="obs", freq="D")
    assert plot is not None


def test_timeseries_plot_plot(clear_figures, sample_data):
    """Test TimeSeriesPlot plot method."""
    plot = TimeSeriesPlot(data=sample_data, x="time", y="obs", freq="D")
    ax = plot.plot()
    assert ax is not None


def test_timeseries_stats_plot(clear_figures, sample_data):
    """Test TimeSeriesStatsPlot plot method."""
    sample_data["model"] = sample_data["obs"] + np.random.rand(10) * 0.1
    plot = TimeSeriesStatsPlot(data=sample_data, col1="obs", col2="model")
    ax = plot.plot(stat="bias", freq="D")
    assert ax is not None
