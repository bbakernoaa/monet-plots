"""
Spatial Track Plot
==================

**What it's for:**
The `SpatialTrack` class visualizes a variable along a geographic path, such
as a flight track, a ship route, or a pollutant plume trajectory.

**When to use:**
Use this when you have observations or model output collected along a moving
platform and want to see how a measured variable changes geographically.

**How to read:**
*   **Path:** The line of points shows the movement of the platform.
*   **Color:** The color of each point represents the value of the variable
    being measured at that location.
*   **Map Features:** Provide geographic context for the path.
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from monet_plots.plots.spatial import SpatialTrack

# 1. Create dummy trajectory data
n_points = 100
time = np.arange(n_points)
lat = 35 + 10 * np.sin(np.linspace(0, np.pi, n_points))
lon = -120 + 40 * np.linspace(0, 1, n_points)
ozone = 20 + 40 * np.random.rand(n_points)

da = xr.DataArray(
    ozone,
    coords={"time": time, "lat": ("time", lat), "lon": ("time", lon)},
    dims=["time"],
    name="ozone",
)

# 2. Initialize and plot
# SpatialTrack identifies lat/lon coordinates automatically
plot = SpatialTrack(da, figsize=(12, 6), states=True, coastlines=True)
plot.plot(cmap="viridis", s=50, edgecolor="black", linewidth=0.5)

# 3. Add title
plot.ax.set_title("Ozone Concentration along Flight Track")
plt.show()
