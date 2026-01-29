"""
Spatial Contour Plot
====================

**What it's for:**
A Spatial Contour plot visualizes continuous 2D fields (like temperature, pressure, or
pollutant concentrations) as line or filled contours on a map.

**When to use:**
Use this to show the spatial distribution and gradients of a variable over a
geographical region. It is ideal for identifying centers of high/low values and
observing the overall structure of a field.

**How to read:**
*   **Contours:** Lines (or boundaries between colors) represent points of equal value.
*   **Color Scale:** Indicates the magnitude of the variable in each region.
*   **Interpretation:** Closely spaced contour lines indicate a steep gradient
    (rapid change over distance), while widely spaced lines indicate a flat field.
"""

import numpy as np
import matplotlib.pyplot as plt
from monet_plots.plots.spatial_contour import SpatialContourPlot


# 1. Create dummy data
class GridObj:
    def __init__(self, lat, lon):
        self.variables = {
            "LAT": lat[np.newaxis, np.newaxis, :, :],
            "LON": lon[np.newaxis, np.newaxis, :, :],
        }


lats = np.linspace(30, 50, 100)
lons = np.linspace(-125, -70, 100)
lon_grid, lat_grid = np.meshgrid(lons, lats)
gridobj = GridObj(lat_grid, lon_grid)

# Create a 2D variable to contour
data = np.sin(lat_grid / 10.0) * np.cos(lon_grid / 10.0)

# 2. Initialize and plot
plot = SpatialContourPlot(data, gridobj, figsize=(10, 8))
plot.plot(levels=15, cmap="viridis")

plot.ax.set_title("Spatial Contour Example")
plt.show()
