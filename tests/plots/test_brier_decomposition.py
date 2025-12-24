import pytest
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from monet_plots.plots.brier_decomposition import BrierScoreDecompositionPlot


@pytest.fixture
def clear_figures():
    """Clear all existing figures before and after a test."""
    plt.close("all")
    yield
    plt.close("all")


@pytest.fixture
def sample_data():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({"forecasts": np.random.rand(100), "observations": np.random.randint(0, 2, 100)})


def test_brier_decomposition_plot(clear_figures, sample_data):
    """Test BrierScoreDecompositionPlot."""
    plot = BrierScoreDecompositionPlot()
    plot.plot(data=sample_data, forecasts_col="forecasts", observations_col="observations")
    assert plot.ax is not None
