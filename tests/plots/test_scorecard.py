import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from monet_plots.plots.scorecard import ScorecardPlot


@pytest.fixture
def clear_figures():
    """Clear all existing figures before and after a test."""
    plt.close("all")
    yield
    plt.close("all")


@pytest.fixture
def sample_data():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame(
        {"x": ["a", "b", "a", "b"], "y": ["c", "c", "d", "d"], "val": np.random.rand(4)}
    )


def test_scorecard_plot(clear_figures, sample_data):
    """Test ScorecardPlot."""
    plot = ScorecardPlot()
    plot.plot(data=sample_data, x_col="x", y_col="y", val_col="val")
    assert plot.ax is not None


def test_scorecard_weathermesh_features(clear_figures):
    """Test new WeatherMesh features in ScorecardPlot."""
    data = pd.DataFrame(
        {
            "city": ["Atlanta", "Boston"],
            "lt": [1, 1],
            "diff": [-0.5, 0.5],
            "mod": [2.5, 3.5],
            "obs": [3.0, 3.0],
        }
    )
    plot = ScorecardPlot()
    plot.plot(
        data,
        x_col="lt",
        y_col="city",
        val_col="diff",
        annot_cols=["mod", "obs"],
        cbar_labels=("Better", "Worse"),
        key_text="Mod | Obs",
    )
    assert plot.ax is not None
    # Check if annotations combined correctly
    # The annotations are in the artists of the axes or in the heatmap data
    # Heatmap puts annotations as text objects
    texts = [t.get_text() for t in plot.ax.texts]
    assert "2.5 | 3.0" in texts
    assert "3.5 | 3.0" in texts


def test_journal_scorecard_groups_cells_and_labels_missing(clear_figures, tmp_path):
    frame = pd.DataFrame(
        {
            "region": ["East", "East", "East", "West"],
            "metric": ["Temperature", "Temperature", "Wind", "Temperature"],
            "period": ["Day 1", "Day 2", "Day 1", "Day 1"],
            "lead": ["00 Z", "00 Z", "00 Z", "00 Z"],
            "change": [-8.0, 3.0, -2.0, 1.0],
            "significant": [True, False, False, True],
        }
    )
    plot = ScorecardPlot(style="journal", figsize=(7, 4), layout="constrained")
    plot.plot(
        frame,
        "lead",
        "metric",
        "change",
        sig_col="significant",
        row_group="region",
        col_group="period",
        baseline="Control forecast",
        units="%",
        lower_is_better=True,
    )
    assert plot.ax.collections[0].get_array().shape == (3, 2)
    assert plot.ax.collections[0].norm.vmin == -8
    assert plot.ax.collections[0].norm.vmax == 8
    assert "Control forecast" in plot.ax.get_title(loc="left")
    assert "−8.0*" in [item.get_text() for item in plot.ax.texts]
    assert sum(item.get_text() == "—" for item in plot.ax.texts) == 2
    plot.save(tmp_path / "scorecard.pdf")
