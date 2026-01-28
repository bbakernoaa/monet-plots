"""
Pivotal Weather Style Example
=============================

This example demonstrates how to use the 'pivotal_weather' style to create a
spatial plot that mimics the aesthetic of Pivotal Weather maps.
"""
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import monet_plots as mplots
from monet_plots.plots.spatial import SpatialPlot

# Set the custom style
mplots.set_style("pivotal_weather")

# Create a sample xarray DataArray
data = np.random.rand(100, 100) * 15
lat = np.linspace(25, 50, 100)
lon = np.linspace(-125, -70, 100)
da = xr.DataArray(
    data,
    coords={"lat": lat, "lon": lon},
    dims=["lat", "lon"],
    name="precipitation",
    attrs={"units": "in"},
)

# Create the spatial plot
fig, ax = plt.subplots(figsize=(12, 8))
splot = SpatialPlot(ax=ax, data=da)

# Plot the data using a pcolormesh
splot.ax.pcolormesh(
    da.lon,
    da.lat,
    da,
    cmap="viridis",
    vmin=0,
    vmax=15,
)

# Add map features
splot.add_features(states=True, coastlines=True, borders=True)

# Add a colorbar
cbar = mplots.colorbar_index(
    ncolors=20,
    cmap="viridis",
    ax=splot.ax,
    orientation="horizontal",
    pad=0.05,
    fraction=0.02,
)
cbar.set_label("Precipitation (in)")

plt.show()
