import pytest
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from monet_plots.plots.taylor_diagram import TaylorDiagramPlot


@pytest.fixture
def clear_figures():
    """Clear all existing figures before and after a test."""
    plt.close("all")
    yield
    plt.close("all")


@pytest.fixture
def sample_taylor_data():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({"obs": np.linspace(0, 10, 100), "model": np.linspace(0, 10, 100) + np.random.rand(100)})


def test_taylor_diagram_plot_creates_plot(clear_figures, sample_taylor_data):
    """Test that TaylorDiagramPlot creates a plot."""
    plot = TaylorDiagramPlot(df=sample_taylor_data, col1="obs", col2="model")
    dia = plot.plot()
    assert dia is not None
