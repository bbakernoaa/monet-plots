"""
Spatial Contour Plot
====================

This example demonstrates how to create a spatial contour plot using the
:class:`monet_plots.plots.spatial_contour.SpatialContour` class.

This plot is ideal for showing variables with smooth gradients, like pressure
or temperature fields.
"""
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

from monet_plots.plots.spatial_contour import SpatialContour
from data import create_dataset

# 1. Create the synthetic dataset
ds = create_dataset()

# 2. Select data for plotting
# We will use the temperature at the 850 hPa level
temp_850 = ds["temperature"].sel(level=850).isel(time=0)

# 3. Create a SpatialContour instance and plot the data
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": ccrs.LambertConformal()})
plot = SpatialContour(ax=ax, data=temp_850, longitude=ds.longitude, latitude=ds.latitude)

# 4. Plot the contours
plot.contour(
    cbar_kwargs={"label": f"850 hPa Temperature ({temp_850.attrs['units']})"}
)

# 5. Add geographic features
plot.add_features("coastline", "states", "countries")

# 6. Set a title
ax.set_title("850 hPa Air Temperature Contours")

plt.show()
