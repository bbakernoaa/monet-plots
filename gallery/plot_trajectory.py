"""
Trajectory Plot
===============

**What it's for:**
A Trajectory plot visualizes the path of a parcel of air or a mobile platform (e.g., aircraft,
ship, or research balloon) over time. It typically combines a horizontal map view of the path
with a time-series view of variables measured along that path.

**When to use:**
Use this when you need to understand the history of an air mass (back-trajectories) or
visualize data collected by mobile sensors. It is essential for source-receptor
analysis in air quality studies.

**How to read:**
*   **Map View (Top/Main):** Shows the geographic path (Longitude/Latitude). Markers or
    colors along the path often represent time or a measured variable.
*   **Time-Series View (Bottom):** Shows how one or more variables changed as the
    platform moved along the trajectory.
*   **Interpretation:** Allows you to correlate specific geographical locations or
    events along the path with observed changes in the measured data.
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
