import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from monet_plots.plots.timeseries import TimeSeriesPlot


def test_timeseries_plot_y2():
    """Test TimeSeriesPlot with secondary y-axis."""
    dates = pd.date_range("2023-01-01", periods=10, freq="h")
    obs = np.random.rand(10)
    alt = np.linspace(0, 1000, 10)

    df = pd.DataFrame({"time": dates, "obs": obs, "altitude": alt})

    plot = TimeSeriesPlot(df, x="time", y="obs")
    ax = plot.plot(y2="altitude", y2_label="Alt (m)", y2_kwargs={"color": "red"})

    assert ax is not None
    # Check for two y-axes (ax and ax.twinx())
    assert len(plot.fig.axes) == 2
    assert plot.fig.axes[1].get_ylabel() == "Alt (m)"
    plt.close(plot.fig)
