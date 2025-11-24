import pytest
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from monet_plots.plots.scorecard import ScorecardPlot

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
        'x': ['a', 'b', 'a', 'b'],
        'y': ['c', 'c', 'd', 'd'],
        'val': np.random.rand(4)
    })

def test_scorecard_plot(clear_figures, sample_data):
    """Test ScorecardPlot."""
    plot = ScorecardPlot()
    plot.plot(data=sample_data, x_col='x', y_col='y', val_col='val')
    assert plot.ax is not None