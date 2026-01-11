"""
Spatial Plot
============

This example demonstrates how to create a basic spatial contour plot using the
:class:`monet_plots.plots.spatial.SpatialPlot` class.

The plot shows the surface air temperature from the synthetic dataset.
"""
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

from monet_plots.plots.spatial_contour import SpatialContourPlot
from data import create_dataset

# 1. Create the synthetic dataset
ds = create_dataset()

# 2. Select data for plotting (first time step, lowest level)
temp_surface = ds["temperature"].isel(time=0, level=0)


# 3. Create a grid object for the plot
class Grid:
    def __init__(self, ds):
        self.variables = {"LAT": ds.latitude, "LON": ds.longitude}


grid = Grid(ds)

# 4. Create a SpatialContourPlot instance and plot the data
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": ccrs.PlateCarree()})
plot = SpatialContourPlot(
    modelvar=temp_surface, gridobj=grid, ax=ax, fig=fig, discrete=False
)

# 5. Plot the data as a filled contour plot
plot.plot(
    levels=15,
    cmap="viridis",
)

# 6. Add geographic features for context
plot.add_features(coastline=True, states=True, countries=True)

# 6. Set a title
ax.set_title("Surface Air Temperature")

# Let mkdocs-gallery display the plot
plt.show()
