# tests/plots/test_curtain.py
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from monet_plots.plots.curtain import CurtainPlot


def test_curtain_plot():
    times = pd.date_range("2023-01-01", periods=10, freq="h")
    levels = np.arange(5)
    data = np.random.rand(5, 10)
    da = xr.DataArray(data, coords=[levels, times], dims=["level", "time"], name="conc")

    plot = CurtainPlot(da)
    ax = plot.plot(kind='pcolormesh')
    assert isinstance(ax, plt.Axes)
    assert ax.get_xlabel() == "time"
    assert ax.get_ylabel() == "level"
    plt.close(plot.fig)


def test_curtain_plot_contourf():
    times = np.arange(10)
    levels = np.arange(5)
    data = np.random.rand(5, 10)
    da = xr.DataArray(data, coords=[levels, times], dims=["level", "dist"], name="conc")

    plot = CurtainPlot(da, x='dist', y='level')
    ax = plot.plot(kind='contourf')
    assert isinstance(ax, plt.Axes)
    plt.close(plot.fig)
