# tests/plots/test_diurnal_error.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from monet_plots.plots.diurnal_error import DiurnalErrorPlot


def test_diurnal_error_plot():
    dates = pd.date_range("2023-01-01", periods=100, freq="h")
    df = pd.DataFrame(
        {
            "time": dates,
            "obs": np.random.rand(100) * 10,
            "mod": np.random.rand(100) * 10,
        }
    )

    # Default (month)
    plot = DiurnalErrorPlot(df, obs_col="obs", mod_col="mod")
    ax = plot.plot()
    assert isinstance(ax, plt.Axes)
    plt.close(plot.fig)


def test_diurnal_error_plot_dayofweek():
    dates = pd.date_range("2023-01-01", periods=500, freq="h")
    df = pd.DataFrame(
        {"time": dates, "obs": np.random.rand(500), "mod": np.random.rand(500)}
    )

    plot = DiurnalErrorPlot(
        df, obs_col="obs", mod_col="mod", second_dim="dayofweek", metric="error"
    )
    ax = plot.plot()
    assert isinstance(ax, plt.Axes)
    assert ax.get_ylabel() == "Day of Week"
    plt.close(plot.fig)
