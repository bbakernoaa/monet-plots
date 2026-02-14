import pytest
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from monet_plots.plots.performance_diagram import PerformanceDiagramPlot


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
        {"pod": np.random.rand(10), "success_ratio": np.random.rand(10)}
    )


def test_performance_diagram_plot(clear_figures, sample_data):
    """Test PerformanceDiagramPlot."""
    plot = PerformanceDiagramPlot(data=sample_data, x_col="success_ratio", y_col="pod")
    plot.plot()
    assert plot.ax is not None
