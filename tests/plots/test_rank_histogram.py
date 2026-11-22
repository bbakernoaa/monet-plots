import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from monet_plots.plots.rank_histogram import RankHistogramPlot


@pytest.fixture
def clear_figures():
    """Clear all existing figures before and after a test."""
    plt.close("all")
    yield
    plt.close("all")


@pytest.fixture
def sample_data():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({"rank": np.random.randint(0, 11, 100)})


def test_rank_histogram_plot(clear_figures, sample_data):
    """Test RankHistogramPlot."""
    plot = RankHistogramPlot()
    plot.plot(data=sample_data, rank_col="rank")
    assert plot.ax is not None


def test_journal_rank_facets_share_scale_and_show_sample_count(tmp_path, clear_figures):
    frame = pd.DataFrame(
        {"rank": [0, 0, 2, 0, 1, 1], "scenario": ["A"] * 3 + ["B"] * 3}
    )
    plot = RankHistogramPlot(style="journal")
    plot.plot(frame, n_members=2, facet_col="scenario")
    first, second = plot.fig.axes
    assert first.get_ylim() == second.get_ylim()
    assert first.get_title(loc="left") == "A · n=3"
    np.testing.assert_allclose(
        [bar.get_height() for bar in first.patches], [2 / 3, 0, 1 / 3]
    )
    plot.save(tmp_path / "rank.pdf")
    assert (tmp_path / "rank.pdf").exists()
