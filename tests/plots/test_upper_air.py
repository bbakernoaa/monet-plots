import pytest
import matplotlib.pyplot as plt
import numpy as np
from monet_plots.plots.upper_air import UpperAir


@pytest.fixture
def clear_figures():
    """Clear all existing figures before and after a test."""
    plt.close('all')
    yield
    plt.close('all')


@pytest.fixture
def sample_data():
    """Create a sample dataset for testing."""
    return {
        'lat': np.linspace(20, 50, 10),
        'lon': np.linspace(-120, -70, 10),
        'hgt': np.random.rand(10, 10),
        'u': np.random.rand(10, 10),
        'v': np.random.rand(10, 10),
    }


def test_upper_air_plot_creates_plot(clear_figures, sample_data):
    """Test that UpperAir creates a plot."""
    plot = UpperAir(**sample_data)
    plot.plot()
    assert plot.ax is not None
