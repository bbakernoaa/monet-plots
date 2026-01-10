# Spatial Plots

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
