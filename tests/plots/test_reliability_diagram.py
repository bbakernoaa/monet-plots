import numpy as np
import pandas as pd
from monet_plots.plots.reliability_diagram import ReliabilityDiagramPlot


def test_reliability_diagram_plot_pre_binned():
    """Test the ReliabilityDiagramPlot plot method with pre-binned data."""
    data = pd.DataFrame({"prob": np.linspace(0, 1, 11), "freq": np.linspace(0, 1, 11)})
    plot = ReliabilityDiagramPlot(data)
    plot.plot()
    assert plot.ax is not None


def test_reliability_diagram_plot_raw_data():
    """Test the ReliabilityDiagramPlot plot method with raw data."""
    data = pd.DataFrame(
        {"forecasts": np.random.rand(100), "observations": np.random.randint(0, 2, 100)}
    )
    plot = ReliabilityDiagramPlot(
        data, forecasts_col="forecasts", observations_col="observations"
    )
    plot.plot()
    assert plot.ax is not None


def test_reliability_diagram_plot_show_hist():
    """Test the ReliabilityDiagramPlot plot method with show_hist=True."""
    data = pd.DataFrame(
        {
            "prob": np.linspace(0, 1, 11),
            "freq": np.linspace(0, 1, 11),
            "count": np.random.randint(1, 100, 11),
        }
    )
    plot = ReliabilityDiagramPlot(data, show_hist=True)
    plot.plot()
    assert plot.ax is not None
