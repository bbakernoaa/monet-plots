"""
Profile Plot
============

This example demonstrates how to create a vertical profile plot using the
:class:`monet_plots.plots.profile.Profile` class.

Profile plots are commonly used in atmospheric science to show how a variable
(like temperature or wind speed) changes with height or pressure level.
"""
import matplotlib.pyplot as plt
import pandas as pd

from monet_plots.plots.profile import Profile
from data import create_dataset

# 1. Create the synthetic dataset
ds = create_dataset()

# 2. Select data for a single vertical profile
# Extract temperature and wind speed for the first time step and a single
# lat/lon point. Convert to a pandas DataFrame.
da_temp = ds["temperature"].isel(time=0, latitude=5, longitude=5)
da_u = ds["u_wind"].isel(time=0, latitude=5, longitude=5)
da_v = ds["v_wind"].isel(time=0, latitude=5, longitude=5)

df = pd.DataFrame({
    'temperature': da_temp.values,
    'u_wind': da_u.values,
    'v_wind': da_v.values,
    'level': da_temp['level'].values
})

# 3. Create a Profile plot instance and plot the data
fig, ax = plt.subplots(figsize=(8, 10))
plot = Profile(ax=ax, data=df)
plot.plot(
    columns=["temperature"],
    y="level",  # Specify the vertical coordinate
    label="Temperature",
    color="red",
    marker="o",
)

# 4. Customize the plot
ax.set_title("Vertical Temperature Profile")
ax.set_ylabel(f"Pressure Level ({ds['level'].attrs['units']})")
ax.set_xlabel(f"Temperature ({ds['temperature'].attrs['units']})")
ax.grid(True, linestyle="--", alpha=0.6)

# Invert the y-axis since pressure decreases with altitude
ax.invert_yaxis()

# Let mkdocs-gallery display the plot
plt.show()
