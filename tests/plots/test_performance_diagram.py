import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from monet_plots.plots.performance_diagram import PerformanceDiagramPlot


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
        {"pod": np.random.rand(10), "success_ratio": np.random.rand(10)}
    )


def test_performance_diagram_plot(clear_figures, sample_data):
    """Test PerformanceDiagramPlot."""
    plot = PerformanceDiagramPlot()
    plot.plot(data=sample_data, x_col="success_ratio", y_col="pod")
    assert plot.ax is not None


def test_journal_performance_identifies_models_and_csi(clear_figures):
    frame = pd.DataFrame(
        {"success_ratio": [0.5, 0.8], "pod": [0.5, 0.8], "model": ["A", "B"]}
    )
    plot = PerformanceDiagramPlot(style="journal")
    plot.plot(frame, label_col="model")
    first, second = plot.ax.lines[1:]
    assert first.get_marker() != second.get_marker()
    assert "CSI 0.33" in first.get_label()
    assert "CSI 0.67" in second.get_label()
    assert "Frequency bias" in plot.ax.get_title(loc="left")
