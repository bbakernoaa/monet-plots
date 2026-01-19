import pytest
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from monet_plots.plots.sp_scatter_bias import SpScatterBiasPlot


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
            "latitude": np.random.uniform(20, 50, 10),
            "longitude": np.random.uniform(-120, -70, 10),
            "col1": np.random.rand(10),
            "col2": np.random.rand(10),
        }
    )


def test_sp_scatter_bias_plot(clear_figures, sample_data):
    """Test SpScatterBiasPlot."""
    plot = SpScatterBiasPlot(df=sample_data, col1="col1", col2="col2")
    ax = plot.plot()
    assert ax is not None
