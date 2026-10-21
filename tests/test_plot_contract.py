"""Library-wide composition and style contracts."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from monet_plots import ScatterPlot, SpatialImshowPlot, SpatialPlot, style_context
from monet_plots.plots.base import BasePlot
from monet_plots.plots.curtain import CurtainPlot
from monet_plots.plots.profile import VerticalProfilePlot
from monet_plots.style import get_style_setting


def test_styles_restore_rcparams_and_custom_settings_on_failure():
    before = plt.rcParams.copy()
    feature_width = get_style_setting("coastline.width")
    with pytest.raises(RuntimeError), style_context("pivotal_weather"):
        assert get_style_setting("coastline.width") == 0.5
        raise RuntimeError("failed render")
    assert dict(plt.rcParams) == dict(before)
    assert get_style_setting("coastline.width") == feature_width


@pytest.mark.parametrize("style", ["wiley", "presentation", None])
def test_plot_construction_and_rendering_do_not_change_session_style(style):
    before = plt.rcParams.copy()
    frame = pd.DataFrame({"x": [0, 1], "y": [1, 2]})
    result = ScatterPlot(data=frame, x="x", y="y", style=style)
    result.plot()
    assert dict(plt.rcParams) == dict(before)
    result.close()


def test_existing_axes_reused_and_figure_mismatch_rejected():
    fig, ax = plt.subplots()
    other = plt.figure()
    with pytest.raises(ValueError, match="owns ax"):
        BasePlot(fig=other, ax=ax)
    plot = BasePlot(ax=ax)
    assert plot.fig is fig
    assert plot.ax is ax
    assert plot.axes[0, 0] is ax
    plt.close("all")


@pytest.mark.parametrize("plot_type", ["image", "curtain", "profile"])
def test_plot_history_is_private_and_lazy(plot_type):
    data = xr.DataArray(
        np.arange(12).reshape(3, 4),
        dims=("lat", "lon"),
        coords={"lat": [30, 35, 40], "lon": [-110, -100, -90, -80]},
        attrs={"history": "original", "units": "ppb"},
    ).chunk({"lat": 2})
    if plot_type == "image":
        plot = SpatialImshowPlot(data, coastlines=False)
        owned = plot.modelvar
    elif plot_type == "curtain":
        plot = CurtainPlot(data)
        owned = plot.data
    else:
        plot = VerticalProfilePlot(data, obs_col="obs", mod_cols="model")
        owned = plot.data
    assert data.attrs == {"history": "original", "units": "ppb"}
    assert owned.attrs["history"] != "original"
    assert hasattr(owned.data, "dask")
    plot.close()


def test_features_use_local_style_without_global_change():
    before = plt.rcParams.copy()
    plot = SpatialPlot(style="pivotal_weather", coastlines=True)
    assert plot.ax.collections[0].get_linewidth()[0] == 0.5
    assert dict(plt.rcParams) == dict(before)
    plot.close()
