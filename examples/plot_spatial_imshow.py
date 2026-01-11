"""
Spatial Imshow Plot
===================

This example demonstrates how to create a spatial plot using `imshow` with the
:class:`monet_plots.plots.spatial_imshow.SpatialImshow` class.

This type of plot is highly efficient for displaying regularly gridded data,
such as satellite imagery or model output on its native grid.
"""
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

from monet_plots.plots.spatial_imshow import SpatialImshow
from data import create_dataset

# 1. Create the synthetic dataset
ds = create_dataset()

# 2. Select data for plotting
# We will use the surface ozone concentration
ozone_surface = ds["ozone"].isel(time=0, level=0)

# 3. Create a SpatialImshow instance and plot the data
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": ccrs.Mercator()})

# The `extent` is automatically inferred from the data's longitude and latitude
# coordinates.
plot = SpatialImshow(
    ax=ax,
    data=ozone_surface,
    longitude=ds.longitude,
    latitude=ds.latitude,
)

# 4. Plot the data using imshow
# The `transform` argument is crucial to tell cartopy that the data's
# coordinates are in a PlateCarree (lat/lon) projection.
plot.imshow(
    cbar_kwargs={"label": f"Surface Ozone ({ozone_surface.attrs['units']})"}
)

# 5. Add geographic features
plot.add_features("coastline", "states", "countries")

# 6. Set a title
ax.set_title("Surface Ozone Concentration")

plt.show()
