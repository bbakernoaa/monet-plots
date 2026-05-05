import matplotlib.pyplot as plt
import numpy as np
import pytest

from monet_plots.plots.ensemble import SpreadSkillPlot


@pytest.fixture
def clear_figures():
    """Clear all existing figures before and after a test."""
    plt.close("all")
    yield
    plt.close("all")


@pytest.fixture
def sample_data():
    """Create a sample DataFrame for testing."""
    return np.random.rand(10), np.random.rand(10)


def test_spread_skill_plot(clear_figures, sample_data):
    """Test SpreadSkillPlot."""
    spread, skill = sample_data
    plot = SpreadSkillPlot(spread=spread, skill=skill)
    ax = plot.plot()
    assert ax is not None
