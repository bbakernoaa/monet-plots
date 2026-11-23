import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

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


def test_dataframe_mean_and_spread_draw_without_pandas(clear_figures, monkeypatch):
    from pandas.plotting import PlotAccessor

    def reject_pandas_plot(*args, **kwargs):
        raise AssertionError("TimeSeriesPlot delegated rendering to pandas")

    monkeypatch.setattr(PlotAccessor, "__call__", reject_pandas_plot)
    dates = pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC")
    frame = pd.DataFrame(
        {"time": dates.repeat(2), "obs": pd.array([1, 3, 2, 6], dtype="Float64")}
    )
    ax = TimeSeriesPlot(frame).plot()
    (line,) = ax.get_lines()
    np.testing.assert_allclose(line.get_ydata(), [2, 4])
    assert list(line.get_xdata()) == list(dates)
    assert ax.collections[0].get_label() == "Mean ± 1 standard deviation"
