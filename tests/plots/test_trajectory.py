import pytest
import numpy as np
from monet_plots.plots import TrajectoryPlot

import pandas as pd


def test_TrajectoryPlot():
    lon = np.linspace(-120, -80, 10)
    lat = np.linspace(30, 40, 10)
    data = np.random.rand(10)
    time = pd.to_datetime(np.arange(10), unit="D")
    ts_data = np.random.rand(10)
    df = pd.DataFrame({"time": time, "value": ts_data})
    plot = TrajectoryPlot(lon, lat, data, df, ts_data)
    plot.plot(freq="1D")
