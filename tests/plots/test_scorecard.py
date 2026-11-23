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


@pytest.mark.parametrize("style", ["journal", "wiley"])
def test_scorecard_accepts_nullable_scores(clear_figures, style):
    frame = pd.DataFrame(
        {
            "lead": [1, 1],
            "region": ["West", "East"],
            "score": pd.Series([1.0, pd.NA], dtype="Float64"),
        }
    )
    plot = ScorecardPlot(style=style)
    plot.plot(frame, "lead", "region", "score")
    values = plot.ax.collections[0].get_array()
    assert values.mask.sum() == 1


def test_grouped_scorecard_accepts_unscaled_norm(clear_figures, sample_data):
    from matplotlib.colors import Normalize

    plot = ScorecardPlot(style="journal")
    norm = Normalize()
    plot.plot(sample_data, "x", "y", "val", norm=norm)
    assert norm.vmin == sample_data.val.min()
    assert norm.vmax == sample_data.val.max()


def test_grouped_scorecard_applies_labels_key_and_significance_without_numbers(
    clear_figures,
):
    data = pd.DataFrame(
        {
            "lead": [1, 1, 1],
            "cycle": ["Morning", "Evening", "Evening"],
            "metric": ["RMSE", "RMSE", "RMSE"],
            "region": ["West", "West", "East"],
            "score": [-2.0, 3.0, float("nan")],
            "significant": [True, False, True],
        }
    )
    plot = ScorecardPlot(style="journal")
    plot.plot(
        data,
        "lead",
        "metric",
        "score",
        row_group="region",
        col_group="cycle",
        cbar_labels=("Better", "Worse"),
        key_text="Difference from control",
        sig_col="significant",
        annot=False,
    )
    assert plot.ax.collections[0].get_array().shape == (2, 2)
    assert [text.get_text() for text in plot.ax.texts].count("*") == 1
    assert plot.ax.child_axes[0].get_xticklabels()[0].get_text() == "Morning"
    assert "Difference from control" in [text.get_text() for text in plot.ax.texts]
    colorbar = plot.ax.collections[0].colorbar
    assert {"Better", "Worse"} <= {text.get_text() for text in colorbar.ax.texts}


def test_scorecard_masks_cells_before_selecting_scale_and_annotations(clear_figures):
    frame = pd.DataFrame(
        {
            "lead": [1, 2, 1, 2],
            "region": ["East", "East", "West", "West"],
            "score": [-1.0, 99.0, 1.0, 4.0],
        }
    )
    plot = ScorecardPlot()
    plot.plot(
        frame,
        "lead",
        "region",
        "score",
        mask=[[False, True], [False, False]],
        fmt="+.0f",
        cbar_kws={"label": "Bias"},
    )
    mesh = plot.ax.collections[0]
    np.testing.assert_array_equal(
        mesh.get_array().mask, [[False, True], [False, False]]
    )
    assert mesh.get_clim() == (-1, 4)
    assert [text.get_text() for text in plot.ax.texts] == ["-1", "+1", "+4"]
    assert mesh.colorbar.ax.get_ylabel() == "Bias"
    assert [label.get_text() for label in plot.ax.get_yticklabels()] == ["East", "West"]
