import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from monet_plots.plots.curtain import CurtainPlot


def test_curtain_plot_overlay():
    """Test CurtainPlot with observation overlay and two-subplot mode."""
    # Create gridded data
    time = np.arange(10)
    level = np.arange(5)
    data = np.random.rand(5, 10)
    da = xr.DataArray(
        data,
        coords={"time": time, "level": level},
        dims=["level", "time"],
        name="model",
    )

    # Create observation data
    obs_time = np.random.choice(time, 20)
    obs_level = np.random.choice(level, 20)
    obs_val = np.random.rand(20)
    obs_ds = xr.Dataset(
        {
            "obs": (["index"], obs_val),
            "time": (["index"], obs_time),
            "level": (["index"], obs_level),
        }
    )

    # Test basic overlay
    plot = CurtainPlot(da, x="time", y="level", obs_data=obs_ds, obs_var="obs")
    ax = plot.plot()
    assert ax is not None
    # Check for scatter overlay (1 collection for contourf, 1 for scatter)
    assert len(ax.collections) >= 2
    plt.close(plot.fig)

    # Test two subplot mode
    plot2 = CurtainPlot(da, x="time", y="level", obs_data=obs_ds, obs_var="obs")
    ax2 = plot2.plot(two_subplot=True)
    assert ax2 is not None
    assert len(plot2.fig.axes) >= 2
    plt.close(plot2.fig)


def test_curtain_plot_pressure_inversion():
    """Test automatic pressure inversion in CurtainPlot."""
    time = np.arange(10)
    pressure = np.linspace(1000, 100, 5)
    da = xr.DataArray(
        np.random.rand(5, 10),
        coords={"time": time, "pressure": pressure},
        dims=["pressure", "time"],
    )

    plot = CurtainPlot(da, x="time", y="pressure")
    ax = plot.plot()
    # Inverted means 1000 at the bottom (lower value on axis), 100 at the top
    # Matplotlib's ylim for inverted axis will have [max, min]
    ylim = ax.get_ylim()
    assert ylim[0] > ylim[1]
    plt.close(plot.fig)
