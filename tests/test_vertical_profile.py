import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from monet_plots.plots.profile import VerticalProfilePlot

try:
    import dask.array as da
except ImportError:
    da = None


def test_vertical_profile_plot_basic():
    """Test basic VerticalProfilePlot with numpy data."""
    alt = np.linspace(0, 10, 100)
    obs = 5 + np.random.randn(100)
    mod = 5.2 + np.random.randn(100)

    data = xr.Dataset(
        {"obs": (["time"], obs), "mod": (["time"], mod), "altitude": (["time"], alt)}
    )

    # Test shading style
    plot = VerticalProfilePlot(
        data, obs_col="obs", mod_cols="mod", interquartile_style="shading"
    )
    ax = plot.plot()
    assert ax is not None
    assert len(ax.get_legend().get_texts()) >= 2
    plt.close(plot.fig)

    # Test box style
    plot_box = VerticalProfilePlot(
        data, obs_col="obs", mod_cols="mod", interquartile_style="box"
    )
    ax_box = plot_box.plot()
    assert ax_box is not None
    plt.close(plot_box.fig)


@pytest.mark.skipif(da is None, reason="dask not installed")
def test_vertical_profile_plot_lazy():
    """Test VerticalProfilePlot with lazy dask data."""
    alt = np.linspace(0, 10, 1000)
    obs = 5 + np.random.randn(1000)
    mod = 5.2 + np.random.randn(1000)

    ds = xr.Dataset(
        {"obs": (["time"], obs), "mod": (["time"], mod), "altitude": (["time"], alt)}
    ).chunk({"time": 250})

    plot = VerticalProfilePlot(ds, obs_col="obs", mod_cols=["mod"])
    ax = plot.plot()
    assert ax is not None
    plt.close(plot.fig)


def test_vertical_profile_supports_pressure_coordinate():
    """Pressure bins plot with larger pressure values toward the bottom."""
    pressure = np.linspace(100, 1000, 200)
    temperature = 25 - 0.07 * (pressure - 100)
    data = xr.Dataset(
        {
            "obs": ("sample", temperature),
            "model": ("sample", temperature + 1),
            "pressure_hpa": ("sample", pressure),
        }
    )
    plot = VerticalProfilePlot(
        data,
        obs_col="obs",
        mod_cols="model",
        alt_col="pressure_hpa",
        bins=np.arange(100, 1101, 100),
        style=None,
    )
    try:
        ax = plot.plot()
        assert ax.yaxis_inverted()
        assert ax.get_ylabel() == "pressure_hpa"
        assert len(ax.lines) >= 2
    finally:
        plot.close()
