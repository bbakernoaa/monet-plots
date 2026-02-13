# Spatial Plots

MONET Plots provides a powerful set of tools for geospatial visualization, supporting both single-panel maps and multi-panel facet grids.

## Xarray Accessor (mplots)

The easiest way to create spatial plots from `xarray` objects is using the `.mplots` accessor. This provides an API similar to xarray's native plotting but automatically adds map features like coastlines and states.

### Examples

```python
import xarray as xr
import monet_plots

# Single panel imshow with coastlines
da.mplots.imshow(coastlines=True)

# Faceted contourf plot by time
da.mplots.contourf(col='time', coastlines=True, states=True)

# Trajectory track plot
da_track.mplots.track(coastlines=True)
```

::: monet_plots.plots.spatial

## SpatialTrack

::: monet_plots.plots.spatial.SpatialTrack

### Example

```python
import numpy as np
from monet_plots.plots import SpatialTrack

# Create sample data
lon = np.linspace(-120, -80, 100)
lat = np.linspace(30, 40, 100)
data = np.random.rand(100)

# Create the plot
plot = SpatialTrack(lon, lat, data)
plot.plot()
```

::: monet_plots.plots.spatial_contour
::: monet_plots.plots.spatial_bias_scatter
