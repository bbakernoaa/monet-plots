import pytest
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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
