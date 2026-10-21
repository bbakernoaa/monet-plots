"""
Pivotal Weather Style
=====================

**What it's for:**
This example demonstrates how to achieve a "Pivotal Weather" aesthetic for
meteorological maps using the built-in `pivotal_weather` style context.

**When to use:**
Use this when you want clean, publication-ready maps with a professional
look similar to popular weather visualization websites. It features a
minimalist map design, a horizontal colorbar at the bottom, and clear
typography.

**How to read:**
*   **Map Context:** States, coastlines, and national borders are clearly
    defined but not distracting.
*   **Colorbar:** Placed horizontally at the bottom to maximize the horizontal
    extent of the map.
*   **Grid:** Usually omitted to keep the focus on the data and geographic
    features.
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import monet_plots as mplots
from monet_plots.plots.spatial_contour import SpatialContourPlot

# 1. Set the custom style
mplots.set_style("pivotal_weather")

# 2. Create sample spatial data
lat = np.linspace(25, 50, 100)
lon = np.linspace(-125, -70, 150)
lon_grid, lat_grid = np.meshgrid(lon, lat)
data = 1000 + 20 * np.sin(lat_grid / 5.0) * np.cos(lon_grid / 10.0)

da = xr.DataArray(
    data,
    coords={"lat": lat, "lon": lon},
    dims=["lat", "lon"],
    name="mslp",
    attrs={"units": "hPa", "long_name": "Mean Sea Level Pressure"},
)

# 3. Initialize and plot
# The 'pivotal_weather' style will automatically influence the map features
# and colorbar placement if used with add_colorbar().
plot = SpatialContourPlot(
    da, figsize=(12, 8), states=True, coastlines=True, borders=True
)

# Plot filled contours
# We can use a specific colormap often associated with pressure or temperature
plot.plot(levels=20, cmap="Spectral_r", filled=True, add_colorbar=True)

# 4. Final touches
plot.ax.set_title(
    "MSLP - Pivotal Weather Aesthetic Example", loc="left", fontweight="bold"
)
plot.ax.set_title("Init: 2023-01-01 00Z", loc="right")

plt.show()
