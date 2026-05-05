import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from monet_plots.plots.scatter import ScatterPlot


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
            "x": np.linspace(0, 10, 100),
            "y": np.linspace(0, 10, 100) + np.random.rand(100),
        }
    )


def test_scatter_plot_creates_plot(clear_figures, sample_data):
    """Test that ScatterPlot creates a plot."""
    plot = ScatterPlot(data=sample_data, x="x", y="y")
    ax = plot.plot()
    assert ax is not None
    assert len(ax.lines) > 0  # Check for regression line
    assert len(ax.collections) > 0  # Check for scatter points


def test_scatter_plot_with_c_and_colorbar(clear_figures, sample_data):
    """Test ScatterPlot with colorization and colorbar."""
    sample_data["c"] = np.random.rand(100)
    plot = ScatterPlot(data=sample_data, x="x", y="y", c="c", colorbar=True)
    ax = plot.plot()
    assert ax is not None
    assert len(ax.figure.axes) > 1  # Check for colorbar axes
