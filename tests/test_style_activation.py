import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from monet_plots.plots.base import BasePlot
from monet_plots.plots.scatter import ScatterPlot
from monet_plots.style import get_available_styles, set_style


def test_get_available_styles():
    styles = get_available_styles()
    assert "wiley" in styles
    assert "paper" in styles
    assert "default" in styles


def test_set_style():
    # Should not raise
    set_style("paper")
    assert plt.rcParams["figure.figsize"] == [8.0, 6.0]

    set_style("wiley")
    assert plt.rcParams["figure.figsize"] == [6.0, 4.0]


def test_base_plot_style_activation():
    # Default should be wiley
    bp = BasePlot()
    assert bp.fig.get_size_inches().tolist() == [6.0, 4.0]
    plt.close(bp.fig)

    # Paper style
    bp2 = BasePlot(style="paper")
    assert bp2.fig.get_size_inches().tolist() == [8.0, 6.0]
    plt.close(bp2.fig)


def test_scatter_plot_style_activation():
    data = pd.DataFrame({"x": np.arange(10), "y": np.arange(10)})

    # Wiley
    sp = ScatterPlot(data=data, x="x", y="y", style="wiley")
    assert sp.fig.get_size_inches().tolist() == [6.0, 4.0]
    plt.close(sp.fig)

    # Presentation
    sp2 = ScatterPlot(data=data, x="x", y="y", style="presentation")
    assert sp2.fig.get_size_inches().tolist() == [12.0, 8.0]
    plt.close(sp2.fig)


def test_spatial_plot_style_activation():
    from monet_plots.plots.spatial import SpatialPlot

    # Paper style for spatial
    # Note: SpatialPlot uses figsize from kwargs if provided, but style should set it if not
    spp = SpatialPlot(style="paper")
    assert spp.fig.get_size_inches().tolist() == [8.0, 6.0]
    plt.close(spp.fig)
