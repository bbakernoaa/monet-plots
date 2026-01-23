import pandas as pd
import pytest
from monet_plots.plots.taylor_diagram import TaylorDiagramPlot


@pytest.fixture
def sample_taylor_data():
    """Create a sample DataFrame for Taylor diagram tests."""
    return pd.DataFrame({"obs": [1.0, 2.0, 3.0], "model": [1.2, 2.3, 3.4]})


def test_taylor_diagram_plot_creates_plot(sample_taylor_data):
    """Test that TaylorDiagramPlot creates a plot and returns an axis."""
    plot = TaylorDiagramPlot(df=sample_taylor_data, col1="obs", col2="model")
    ax = plot.plot()
    assert ax is not None
    assert len(plot.sample_points) == 2  # Reference point + one model


def test_taylor_diagram_multiple_models(sample_taylor_data):
    """Test TaylorDiagramPlot with multiple models."""
    df = sample_taylor_data.copy()
    df["model2"] = [0.8, 1.9, 2.9]
    plot = TaylorDiagramPlot(df=df, col1="obs", col2=["model", "model2"])
    ax = plot.plot()
    assert ax is not None
    assert len(plot.sample_points) == 3  # Reference point + two models


def test_taylor_diagram_axes_labels(sample_taylor_data):
    """Test that the axes labels are set correctly."""
    plot = TaylorDiagramPlot(df=sample_taylor_data, col1="obs", col2="model")
    ax = plot.plot()
    # The container axes is self._ax, which is returned by plot()
    assert ax.axis["top"].label.get_text() == "Correlation"
    assert ax.axis["left"].label.get_text() == "Standard deviation"
