import pytest
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from monet_plots.plots.rank_histogram import RankHistogramPlot


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
        'rank': np.random.randint(0, 11, 100)
    })


def test_rank_histogram_plot(clear_figures, sample_data):
    """Test RankHistogramPlot."""
    plot = RankHistogramPlot()
    plot.plot(data=sample_data, rank_col='rank')
    assert plot.ax is not None
