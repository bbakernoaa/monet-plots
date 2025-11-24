import pytest
import matplotlib.pyplot as plt
import numpy as np
from monet_plots.plots.profile import ProfilePlot

@pytest.fixture
def clear_figures():
    """Clear all existing figures before and after a test."""
    plt.close('all')
    yield
    plt.close('all')

@pytest.fixture
def sample_data_line():
    """Create sample data for a line plot."""
    return {
        'x': np.linspace(0, 10, 100),
        'y': np.linspace(0, 10, 100) + np.random.rand(100)
    }

@pytest.fixture
def sample_data_contour():
    """Create sample data for a contour plot."""
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 10, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X) * np.cos(Y)
    return {'x': X, 'y': Y, 'z': Z}

def test_profile_plot_line_creates_plot(clear_figures, sample_data_line):
    """Test that ProfilePlot creates a line plot."""
    plot = ProfilePlot(**sample_data_line)
    plot.plot()
    assert plot.ax is not None
    assert len(plot.ax.lines) > 0

def test_profile_plot_contour_creates_plot(clear_figures, sample_data_contour):
    """Test that ProfilePlot creates a contour plot."""
    plot = ProfilePlot(**sample_data_contour)
    plot.plot()
    assert plot.ax is not None
    assert len(plot.ax.collections) > 0
