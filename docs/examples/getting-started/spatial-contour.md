# Spatial Contour Plots

Spatial contour plots are a common way to visualize 2D gridded data, such as model output or satellite observations. `monet_plots` provides the `SpatialContourPlot` class to easily create these plots with map projections and geographic features.

## Prerequisites

- Basic Python knowledge
- Understanding of xarray and numpy
- MONET Plots installed (`pip install monet_plots`)
- Matplotlib and Cartopy installed (`pip install matplotlib cartopy`)

## Plotting Workflow

1.  **Prepare Data**: Your data should be in an `xarray.Dataset` or a similar object that contains `LAT` and `LON` coordinate variables. The variable you want to plot should be a 2D array.
2.  **Initialize `SpatialContourPlot`**: Create an instance of the class, passing the data variable and the grid object.
3.  **Call `plot` method**: Generate the contour plot, optionally specifying the number of levels, colormap, etc.
4.  **Add Features**: Use the `add_features` method to add coastlines, states, etc.
5.  **Customize**: Add titles, labels, and other visual enhancements.
6.  **Display/Save**: Show the plot using `plt.show()` or save it to a file.

## Example: Basic Spatial Contour Plot

```python
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from monet_plots.plots.spatial_contour import SpatialContourPlot

# 1. Prepare sample data
lat = np.linspace(25, 50, 50)
lon = np.linspace(-125, -70, 50)
lon2d, lat2d = np.meshgrid(lon, lat)
temp = np.sin(np.deg2rad(lat2d) * 4) + np.cos(np.deg2rad(lon2d) * 4)

ds = xr.Dataset(
    {
        "temperature": (("y", "x"), temp),
        "LAT": (("y", "x"), lat2d),
        "LON": (("y", "x"), lon2d),
    },
    coords={"y": np.arange(50), "x": np.arange(50)},
)

# 2. Initialize and create the plot
fig, ax = plt.subplots(
    figsize=(10, 8), subplot_kw={"projection": ccrs.LambertConformal()}
)

plot = SpatialContourPlot(
    modelvar=ds["temperature"], gridobj=ds, ax=ax, fig=fig, discrete=False
)

# 3. Plot the data
plot.plot(levels=15, cmap="viridis")

# 4. Add geographic features
plot.add_features("coastline", "states", "countries")

# 5. Set title and extent
ax.set_title("Surface Temperature Contour")
ax.set_extent([-125, -70, 25, 50], crs=ccrs.PlateCarree())

plt.tight_layout()
plt.show()
```
