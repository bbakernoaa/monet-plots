import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from monet_plots.plots.roc_curve import ROCCurvePlot


@pytest.fixture
def clear_figures():
    """Clear all existing figures before and after a test."""
    plt.close("all")
    yield
    plt.close("all")


@pytest.fixture
def sample_data():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({"pofd": np.linspace(0, 1, 10), "pod": np.linspace(0, 1, 10)})


def test_roc_curve_plot(clear_figures, sample_data):
    """Test ROCCurvePlot."""
    plot = ROCCurvePlot()
    plot.plot(data=sample_data, x_col="pofd", y_col="pod")
    assert plot.ax is not None
