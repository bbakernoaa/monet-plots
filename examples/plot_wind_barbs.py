"""
Wind Barbs Plot
===============

**What it's for:**
A Wind Barbs plot uses standardized meteorological symbols to represent wind
speed and direction at various points on a map.

**When to use:**
Use this when you need a clear, precise representation of wind speed categories
and directions across a geographic region. It is the traditional way to display
wind data on weather maps.

**How to read:**
*   **The Staff:** Points in the direction the wind is blowing *from*.
*   **The Barbs/Flags:** Attached to the end of the staff. Each full barb
    represents 10 units of speed (e.g., knots or m/s), a half-barb is 5 units,
    and a pennant (flag) is 50 units.
*   **Interpretation:** Allows you to quickly assess wind speed and direction
    patterns, identifying features like wind shifts and areas of high wind speeds.
"""

import matplotlib.pyplot as plt
import numpy as np

from monet_plots.plots.wind_barbs import WindBarbsPlot


# 1. Prepare sample data
class GridObj:
    def __init__(self, lat, lon):
        self.variables = {
            "LAT": lat[np.newaxis, np.newaxis, :, :],
            "LON": lon[np.newaxis, np.newaxis, :, :],
        }


lats = np.linspace(30, 50, 10)
lons = np.linspace(-125, -70, 10)
lon_grid, lat_grid = np.meshgrid(lons, lats)
gridobj = GridObj(lat_grid, lon_grid)

ws = np.random.uniform(5, 50, (10, 10))
wdir = np.random.uniform(0, 360, (10, 10))

# 2. Initialize and plot
plot = WindBarbsPlot(ws, wdir, gridobj, figsize=(10, 8))
plot.plot()

plt.title("Wind Barbs Example")
plt.show()
