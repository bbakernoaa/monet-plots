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
