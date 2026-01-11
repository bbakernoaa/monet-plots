"""
Wind Quiver Plot
================

This example demonstrates how to create a wind quiver plot using the
:class:`monet_plots.plots.wind_quiver.WindQuiver` class.

Quiver plots use arrows to show both the direction and magnitude of a vector
field, making them ideal for visualizing wind.
"""
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

from monet_plots.plots.wind_quiver import WindQuiver
from data import create_dataset

# 1. Create the synthetic dataset
ds = create_dataset()

# 2. Select data for plotting
u_wind = ds["u_wind"].sel(level=500).isel(time=0)
v_wind = ds["v_wind"].sel(level=500).isel(time=0)
speed = np.sqrt(u_wind**2 + v_wind**2)

# 3. Create a WindQuiver instance and plot the data
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": ccrs.LambertConformal()})
plot = WindQuiver(
    ax=ax,
    u=u_wind,
    v=v_wind,
    longitude=ds.longitude,
    latitude=ds.latitude,
)

# 4. Plot the quiver arrows, colored by wind speed
# We can thin the data to avoid overcrowding.
plot.quiver(
    thin=2,
    color=speed,
    cmap="viridis",
    cbar_kwargs={"label": "Wind Speed (m/s)"},
)

# 5. Add geographic features
plot.add_features("coastline", "states", "countries")

# 6. Set a title
ax.set_title("500 hPa Wind Quiver")

plt.show()
