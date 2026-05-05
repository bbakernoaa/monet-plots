import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from monet_plots.plots.conditional_bias import ConditionalBiasPlot


@pytest.fixture
def clear_figures():
    """Clear all existing figures before and after a test."""
    plt.close("all")
    yield
    plt.close("all")


@pytest.fixture
def sample_data():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({"obs": np.random.rand(100), "fcst": np.random.rand(100)})


def test_conditional_bias_plot(clear_figures, sample_data):
    """Test ConditionalBiasPlot."""
    plot = ConditionalBiasPlot()
    plot.plot(data=sample_data, obs_col="obs", fcst_col="fcst")
    assert plot.ax is not None


def test_conditional_bias_plot_with_label_col(clear_figures):
    df = pd.DataFrame(
        {
            "obs": np.random.rand(100),
            "fcst": np.random.rand(100),
            "group": np.random.choice(["A", "B"], size=100),
        }
    )
    plot = ConditionalBiasPlot()
    plot.plot(data=df, obs_col="obs", fcst_col="fcst", label_col="group")
    # Should have a legend for groups
    assert plot.ax.get_legend() is not None


def test_conditional_bias_plot_n_bins(clear_figures, sample_data):
    plot = ConditionalBiasPlot()
    plot.plot(data=sample_data, obs_col="obs", fcst_col="fcst", n_bins=5)
    # Should have 5 or fewer points (bins with >=5 samples)
    lines = plot.ax.get_lines()
    assert any(len(line.get_xdata()) <= 5 for line in lines)


def test_conditional_bias_plot_empty_df(clear_figures):
    df = pd.DataFrame({"obs": [], "fcst": []})
    plot = ConditionalBiasPlot()
    with pytest.raises(ValueError):
        plot.plot(data=df, obs_col="obs", fcst_col="fcst")


def test_conditional_bias_plot_missing_column(clear_figures, sample_data):
    df = sample_data.drop(columns=["obs"])
    plot = ConditionalBiasPlot()
    with pytest.raises(ValueError):
        plot.plot(data=df, obs_col="obs", fcst_col="fcst")


def test_conditional_bias_plot_few_samples_per_bin(clear_figures):
    # Only 1 sample per bin, so no points should be plotted
    df = pd.DataFrame({"obs": np.arange(10), "fcst": np.arange(10) + 1})
    plot = ConditionalBiasPlot()
    plot.plot(data=df, obs_col="obs", fcst_col="fcst", n_bins=10)
    # No error, but no data points (lines) should be present except axhline
    lines = plot.ax.get_lines()
    # Only the zero-bias line should be present
    assert len(lines) == 1


def test_conditional_bias_zero_bias_line(clear_figures, sample_data):
    plot = ConditionalBiasPlot()
    plot.plot(data=sample_data, obs_col="obs", fcst_col="fcst")
    # Check for a horizontal line at y=0
    found = any(
        getattr(line, "get_ydata", lambda: [])()[0] == 0
        and all(y == 0 for y in line.get_ydata())
        for line in plot.ax.get_lines()
    )
    assert found
