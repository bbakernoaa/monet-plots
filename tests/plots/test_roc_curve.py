import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from monet_plots.plots.roc_curve import ROCCurvePlot


@pytest.fixture
def clear_figures():
    """Clear all existing figures before and after a test."""
    plt.close("all")
    yield
    plt.close("all")


@pytest.fixture
def sample_data():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({"pofd": np.linspace(0, 1, 10), "pod": np.linspace(0, 1, 10)})


def test_roc_curve_plot(clear_figures, sample_data):
    """Test ROCCurvePlot."""
    plot = ROCCurvePlot()
    plot.plot(data=sample_data, x_col="pofd", y_col="pod")
    assert plot.ax is not None


def test_journal_roc_styles_curves_without_overlapping_fills(
    clear_figures, sample_data
):
    data = pd.concat([sample_data.assign(model="A"), sample_data.assign(model="B")])
    plot = ROCCurvePlot(style="journal")
    plot.plot(data, label_col="model", title="Event discrimination")
    first, second = plot.ax.lines[1:]
    assert first.get_linestyle() != second.get_linestyle()
    assert not plot.ax.collections
    assert "AUC=0.500" in first.get_label()
    assert plot.ax.get_title(loc="left") == "Event discrimination"


def test_roc_auc_is_invariant_to_row_order_at_vertical_segments(clear_figures):
    points = pd.DataFrame({"pofd": [0, 0, 1], "pod": [0, 0.5, 1]})
    for data in (points, points.iloc[[1, 0, 2]]):
        plot = ROCCurvePlot(style="journal")
        plot.plot(data)
        line = plot.ax.lines[-1]
        np.testing.assert_allclose(line.get_ydata(), [0, 0.5, 1])
        assert "AUC=0.750" in line.get_label()
        plot.close()
