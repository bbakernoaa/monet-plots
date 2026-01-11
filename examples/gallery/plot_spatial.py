"""
Spatial Plot
============

This example demonstrates how to create a basic spatial contour plot using the
:class:`monet_plots.plots.spatial.SpatialPlot` class.

The plot shows the surface air temperature from the synthetic dataset.
"""
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

from monet_plots.plots.spatial import SpatialPlot
from data import create_dataset

# 1. Create the synthetic dataset
ds = create_dataset()

# 2. Select data for plotting (first time step, lowest level)
temp_surface = ds["temperature"].isel(time=0, level=0)

# 3. Create a SpatialPlot instance
# The `projection` argument is crucial for cartopy.
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": ccrs.PlateCarree()})
plot = SpatialPlot(ax=ax, data=temp_surface)

# 4. Plot the data as a filled contour plot
# The `transform` argument tells cartopy the data's original coordinate system.
plot.contourf(
    cbar_kwargs={"label": f"Surface Temperature ({temp_surface.attrs['units']})"}
)

# 5. Add geographic features for context
plot.add_features("coastline", "states", "countries")

# 6. Set a title
ax.set_title("Surface Air Temperature")

# Let mkdocs-gallery display the plot
plt.show()
