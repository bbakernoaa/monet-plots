import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from unittest import mock

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

    # Test pivotal_weather (has custom keys)
    set_style("pivotal_weather")
    from monet_plots.style import get_style_setting

    assert get_style_setting("coastline.width") == 0.7
    assert get_style_setting("font.size") == 11
    assert get_style_setting("ocean")["facecolor"] == "#d7e7f5"
    assert get_style_setting("cbar.location") == "bottom"


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


def test_base_plot_colorbar_uses_pivotal_style_defaults():
    bp = BasePlot(style="pivotal_weather")
    mappable = plt.cm.ScalarMappable(cmap="viridis")

    cb = bp.add_colorbar(mappable, label="Bias")

    assert cb.orientation == "horizontal"
    plt.close(bp.fig)


def test_spatial_plot_pivotal_weather_injects_feature_defaults():
    from monet_plots.plots.spatial import SpatialPlot

    with (
        mock.patch.object(SpatialPlot, "_draw_single_feature") as draw_single_feature,
        mock.patch.object(SpatialPlot, "_draw_gridlines") as draw_gridlines,
    ):
        spp = SpatialPlot(style="pivotal_weather")

    style_args = [call.args[0] for call in draw_single_feature.call_args_list]
    assert True in style_args
    assert any(isinstance(arg, dict) and arg.get("facecolor") == "#d7e7f5" for arg in style_args)
    draw_gridlines.assert_called_once_with(False)
    plt.close(spp.fig)
