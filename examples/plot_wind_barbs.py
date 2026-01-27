"""
Wind Barbs
==========

This example demonstrates how to create a Wind Barbs plot.
"""

import numpy as np
import matplotlib.pyplot as plt
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
