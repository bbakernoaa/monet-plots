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

from monet_plots.plots.spatial_contour import SpatialContourPlot
from data import create_dataset

# 1. Create the synthetic dataset
ds = create_dataset()

# 2. Select data for plotting
# We will use the temperature at the 850 hPa level
temp_850 = ds["temperature"].sel(level=850).isel(time=0)


# 3. Create a grid object for the plot
class Grid:
    def __init__(self, ds):
        self.variables = {"LAT": ds.latitude, "LON": ds.longitude}


grid = Grid(ds)

# 4. Create a SpatialContourPlot instance and plot the data
fig, ax = plt.subplots(
    figsize=(10, 8), subplot_kw={"projection": ccrs.LambertConformal()}
)
plot = SpatialContourPlot(
    modelvar=temp_850, gridobj=grid, ax=ax, fig=fig, discrete=False
)

# 5. Plot the contours
plot.plot(
    levels=15,
    cmap="viridis",
)

# 5. Add geographic features
plot.add_features("coastline", "states", "countries")

# 6. Set a title
ax.set_title("850 hPa Air Temperature Contours")

plt.show()
