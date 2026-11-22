import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from monet_plots.plots.meteogram import Meteogram


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
        {"temp": np.random.rand(10), "pressure": np.random.rand(10)},
        index=pd.to_datetime(np.arange(10), unit="D"),
    )


def test_meteogram_plot_creates_plot(clear_figures, sample_data):
    """Test that Meteogram creates a plot."""
    plot = Meteogram(df=sample_data, variables=["temp", "pressure"])
    plot.plot()
    assert plot.ax is not None


def test_meteogram_uses_shared_concise_time_axis(clear_figures, sample_data):
    from matplotlib.dates import ConciseDateFormatter

    sample_data.index = sample_data.index.tz_localize("UTC")
    plot = Meteogram(df=sample_data, variables=["temp", "pressure"], style="journal")
    plot.plot()
    first, last = plot.ax
    assert first.get_shared_x_axes().joined(first, last)
    assert isinstance(last.xaxis.get_major_formatter(), ConciseDateFormatter)
    assert "UTC" in last.get_xlabel()
