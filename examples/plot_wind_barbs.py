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

from monet_plots.plots.wind_barbs import WindBarbs
from data import create_dataset

# 1. Create the synthetic dataset
ds = create_dataset()

# 2. Select data for plotting
# We'll use the wind components at the 850 hPa level.
u_wind = ds["u_wind"].sel(level=850).isel(time=0)
v_wind = ds["v_wind"].sel(level=850).isel(time=0)

# 3. Create a WindBarbs instance and plot the data
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": ccrs.PlateCarree()})

# The plot class takes u and v components, along with coordinates.
plot = WindBarbs(
    ax=ax,
    u=u_wind,
    v=v_wind,
    longitude=ds.longitude,
    latitude=ds.latitude,
)

# 4. Plot the wind barbs
# We can thin the data to avoid overcrowding the plot using the `thin` arg.
plot.barbs(thin=2, length=7)

# 5. Add geographic features
plot.add_features("coastline", "states", "countries")

# 6. Set a title
ax.set_title("850 hPa Wind Barbs")

plt.show()
