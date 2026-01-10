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


def test_timeseries_plot_init(clear_figures, sample_data):
    """Test TimeSeriesPlot initialization."""
    plot = TimeSeriesPlot(df=sample_data, x="time", y="obs")
    assert plot is not None


def test_timeseries_plot_plot(clear_figures, sample_data):
    """Test TimeSeriesPlot plot method."""
    plot = TimeSeriesPlot(df=sample_data, x="time", y="obs")
    ax = plot.plot()
    assert ax is not None
