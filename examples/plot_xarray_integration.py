"""
Xarray Integration
==================

**What it's for:**
This example demonstrates how MONET Plots integrates directly with `xarray` data
structures.

**When to use:**
Use this whenever your data is already stored in xarray `DataArray` or `Dataset`
formats. MONET Plots will automatically leverage the coordinates and attributes
(like `units`, `long_name`, and `standard_name`) to automate plot labeling and
formatting.

**How to read:**
*   **Axes/Labels:** Notice that the axes labels and plot title are automatically
    populated from the xarray metadata.
*   **Interpretation:** The plot is interpreted according to its specific
    visualization type (in this case, a Time Series), but with significantly
    less manual configuration required.
"""

import xarray as xr
import pandas as pd
import numpy as np
from monet_plots.plots.timeseries import TimeSeriesPlot

# Create sample xarray data
dates = pd.date_range("2023-01-01", periods=100, freq="h")
temperature = (
    15 + 5 * np.sin(np.arange(100) * 2 * np.pi / 24) + np.random.normal(0, 1, 100)
)

da = xr.DataArray(
    temperature,
    dims=["time"],
    coords={"time": dates},
    name="temperature",
    attrs={"units": "°C", "long_name": "Air Temperature"},
)

# Create and plot
plot = TimeSeriesPlot(
    da,
    x="time",
    y="temperature",
    title="Temperature Time Series",
    ylabel="Temperature (°C)",
)
ax = plot.plot()
plot.save("temperature_timeseries.png")
plot.close()
