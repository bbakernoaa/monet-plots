import pytest
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from monet_plots.plots.kde import KDEPlot


@pytest.fixture
def clear_figures():
    """Clear all existing figures before and after a test."""
    plt.close("all")
    yield
    plt.close("all")


@pytest.fixture
def sample_data():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({"x": np.linspace(0, 10, 100), "y": np.linspace(0, 10, 100) + np.random.rand(100)})


def test_kde_plot_creates_plot(clear_figures, sample_data):
    """Test that KDEPlot creates a plot."""
    plot = KDEPlot(df=sample_data, x="x", y="y")
    ax = plot.plot()
    assert ax is not None
