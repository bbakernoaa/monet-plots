# Trajectory Plot

This example shows how to create a trajectory plot, which includes a spatial track on a map and a timeseries plot.

```python
import numpy as np
import pandas as pd
from monet_plots.plots import TrajectoryPlot

# Create sample data
lon = np.linspace(-120, -80, 100)
lat = np.linspace(30, 40, 100)
data = np.random.rand(100)
time = pd.to_datetime(np.arange(100), unit='D')
ts_data = np.random.rand(100)
df = pd.DataFrame({'time': time, 'value': ts_data})

# Create the plot
plot = TrajectoryPlot(lon, lat, data, df, 'value')
plot.plot()
```
