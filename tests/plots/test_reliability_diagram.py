import numpy as np
import pandas as pd

from monet_plots.plots.reliability_diagram import ReliabilityDiagramPlot


def test_reliability_diagram_plot_pre_binned():
    """Test the ReliabilityDiagramPlot plot method with pre-binned data."""
    plot = ReliabilityDiagramPlot()
    data = pd.DataFrame({"prob": np.linspace(0, 1, 11), "freq": np.linspace(0, 1, 11)})
    plot.plot(data)
    assert plot.ax is not None


def test_reliability_diagram_plot_raw_data():
    """Test the ReliabilityDiagramPlot plot method with raw data."""
    plot = ReliabilityDiagramPlot()
    data = pd.DataFrame(
        {"forecasts": np.random.rand(100), "observations": np.random.randint(0, 2, 100)}
    )
    plot.plot(data, forecasts_col="forecasts", observations_col="observations")
    assert plot.ax is not None


def test_reliability_diagram_plot_show_hist():
    """Test the ReliabilityDiagramPlot plot method with show_hist=True."""
    plot = ReliabilityDiagramPlot()
    data = pd.DataFrame(
        {
            "prob": np.linspace(0, 1, 11),
            "freq": np.linspace(0, 1, 11),
            "count": np.random.randint(1, 100, 11),
        }
    )
    plot.plot(data, show_hist=True)
    assert plot.ax is not None


def test_empty_bins_are_gaps_with_correct_skill_boundary():
    plot = ReliabilityDiagramPlot()
    try:
        data = pd.DataFrame(
            {"prob": [0.1, 0.5, 0.9], "freq": [0.2, 0, 0.8], "count": [5, 0, 5]}
        )
        plot.plot(data, climatology=0.4)
        assert np.isnan(plot.ax.lines[-1].get_ydata()[1])
        boundary = next(
            line for line in plot.ax.lines if line.get_label() == "No skill"
        )
        np.testing.assert_allclose(boundary.get_ydata(), [0.2, 0.7])
    finally:
        plot.close()
