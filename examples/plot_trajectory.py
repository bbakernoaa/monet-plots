"""
Trajectory Plot
===============

This example demonstrates how to create a Trajectory plot, combining a map track and a timeseries.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from monet_plots.plots.trajectory import TrajectoryPlot

# 1. Prepare sample data
n_points = 50
time = pd.date_range("2023-01-01", periods=n_points, freq="h")
lat = np.linspace(30, 40, n_points) + np.random.normal(0, 0.5, n_points)
lon = np.linspace(-120, -110, n_points) + np.random.normal(0, 0.5, n_points)
data = np.sin(np.linspace(0, 5, n_points)) * 10 + 20
ts_data = data + np.random.normal(0, 2, n_points)

# 2. Initialize and plot
# TrajectoryPlot(longitude, latitude, data, time, ts_data, ...)
plot = TrajectoryPlot(lon, lat, data, time, ts_data, figsize=(12, 6))
plot.plot()

plt.show()
