"""
Spatial Imshow Plot
==================

**What it's for:**
The `SpatialImshowPlot` class provides an efficient way to visualize 2D gridded
geospatial data using Matplotlib's `imshow` method.

**When to use:**
Use this for visualizing large, regular grids of data where performance is
important and you don't need the contouring of `SpatialContourPlot`. It is
ideal for satellite imagery, model output on regular grids, and radar data.

**How to read:**
*   **Pixels:** Each pixel represents a value in the input grid.
*   **Coordinates:** The axes show longitude and latitude.
*   **Color Scale:** Indicates the magnitude of the variable.
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from monet_plots.plots.spatial_imshow import SpatialImshowPlot

# 1. Create dummy xarray data
lat = np.linspace(25, 50, 100)
lon = np.linspace(-125, -70, 120)
data = np.sin(lat[:, np.newaxis] / 5.0) * np.cos(lon[np.newaxis, :] / 5.0)
da = xr.DataArray(
    data,
    coords={"lat": lat, "lon": lon},
    dims=["lat", "lon"],
    name="ozone",
    attrs={"units": "ppb"},
)

# 2. Initialize and plot
# SpatialImshowPlot handles the map setup and plotting
plot = SpatialImshowPlot(da, figsize=(10, 8), states=True, coastlines=True)
plot.plot(cmap="plasma")

# 3. Add title and colorbar label
plot.ax.set_title("Ozone Concentration (Imshow)")
plt.show()
