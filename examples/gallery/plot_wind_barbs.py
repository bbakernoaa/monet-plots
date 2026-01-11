"""
Wind Barbs Plot
===============

This example demonstrates how to create a map with wind barbs using the
:class:`monet_plots.plots.wind_barbs.WindBarbs` class.

Wind barbs are a standard way to visualize wind speed and direction in
meteorology.
"""
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

import numpy as np
from monet_plots.plots.wind_barbs import WindBarbsPlot
from data import create_dataset

# 1. Create the synthetic dataset
ds = create_dataset()

# 2. Select data for plotting and derive wind speed/direction
# We'll use the wind components at the 850 hPa level.
u_wind = ds["u_wind"].sel(level=850).isel(time=0)
v_wind = ds["v_wind"].sel(level=850).isel(time=0)
ws = np.sqrt(u_wind**2 + v_wind**2)
wdir = 180 / np.pi * np.arctan2(-u_wind, -v_wind)


# 3. Create a grid object for the plot
class Grid:
    def __init__(self, ds):
        self.variables = {"LAT": ds.latitude, "LON": ds.longitude}


grid = Grid(ds)

# 4. Create a WindBarbsPlot instance and plot the data
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": ccrs.PlateCarree()})
plot = WindBarbsPlot(ws=ws, wdir=wdir, gridobj=grid, ax=ax, fig=fig)

# 5. Plot the wind barbs
plot.plot(skip=3, length=7)

# 5. Add geographic features
plot.add_features("coastline", "states", "countries")

# 6. Set a title
ax.set_title("850 hPa Wind Barbs")

plt.show()
