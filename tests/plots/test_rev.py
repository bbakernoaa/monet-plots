import pytest
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from monet_plots.plots.rev import RelativeEconomicValuePlot


@pytest.fixture
def clear_figures():
    """Clear all existing figures before and after a test."""
    plt.close('all')
    yield
    plt.close('all')


@pytest.fixture
def sample_data():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'hits': np.random.randint(0, 100, 10),
        'misses': np.random.randint(0, 100, 10),
        'fa': np.random.randint(0, 100, 10),
        'cn': np.random.randint(0, 100, 10)
    })


def test_rev_plot(clear_figures, sample_data):
    """Test RelativeEconomicValuePlot."""
    plot = RelativeEconomicValuePlot()
    plot.plot(data=sample_data)
    assert plot.ax is not None
