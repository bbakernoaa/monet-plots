import pytest
import matplotlib.pyplot as plt
import numpy as np
from monet_plots.plots.windrose import Windrose


@pytest.fixture
def clear_figures():
    """Clear all existing figures before and after a test."""
    plt.close('all')
    yield
    plt.close('all')


@pytest.fixture
def sample_data():
    """Create a sample DataFrame for testing."""
    return {
        'wd': np.random.uniform(0, 360, 100),
        'ws': np.random.uniform(0, 20, 100)
    }


def test_windrose_plot_creates_plot(clear_figures, sample_data):
    """Test that Windrose creates a plot."""
    plot = Windrose(wd=sample_data['wd'], ws=sample_data['ws'])
    plot.plot()
    assert plot.ax is not None
