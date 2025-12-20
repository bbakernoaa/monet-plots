import pytest
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from monet_plots.plots.facet_grid import FacetGridPlot


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
        'x': np.random.rand(100),
        'y': np.random.rand(100),
        'row': np.random.choice(['a', 'b'], 100),
        'col': np.random.choice(['c', 'd'], 100)
    })


def test_facet_grid_plot(clear_figures, sample_data):
    """Test FacetGridPlot."""
    plot = FacetGridPlot(data=sample_data, row='row', col='col')
    assert plot.grid is not None
