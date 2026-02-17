"""
Facet Grid Spatial
==================

**What it's for:**
Facet Grid Spatial is an extension of the facet grid concept applied to geographic
maps. It allows you to create a matrix of maps, where each map represents a different
subset of your data (e.g., different models and different timestamps).

**When to use:**
Use this when you need to compare spatial patterns across multiple dimensions
simultaneously, such as comparing the output of several different models at
multiple points in time. It is the most effective way to identify where and
when models diverge in their spatial predictions.

**How to read:**
*   **Rows/Columns:** Represent different categorical dimensions (e.g., Time vs. Model).
*   **Subplots:** Each facet is a complete `SpatialPlot` showing the geographic
    distribution of a variable.
*   **Interpretation:** Look for spatial shifts or intensity differences between
    the maps in different rows and columns.
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from monet_plots.plots.facet_grid import SpatialFacetGridPlot
from monet_plots.plots.spatial_imshow import SpatialImshowPlot

# 1. Create some dummy spatial data
# Simulate data for two models ('model_A', 'model_B') and two times ('2023-01-01', '2023-01-02')
lats = np.arange(30, 35, 0.5)
lons = np.arange(-100, -95, 0.5)
times = pd.to_datetime(["2023-01-01", "2023-01-02"])
models = ["model_A", "model_B"]

# Create a DataArray with dimensions (time, model, lat, lon)
data = xr.DataArray(
    np.random.rand(len(times), len(models), len(lats), len(lons)) * 100,
    coords={"time": times, "model": models, "lat": lats, "lon": lons},
    dims=["time", "model", "lat", "lon"],
    name="temperature",
)

# Add some variation for demonstration
data.loc[{"model": "model_A", "time": "2023-01-01"}] += 10
data.loc[{"model": "model_B", "time": "2023-01-02"}] -= 5

# Convert to Dataset if you have multiple variables, or keep as DataArray
ds = data.to_dataset()


# 2. Use SpatialFacetGridPlot
# We want 'time' as rows and 'model' as columns
grid = SpatialFacetGridPlot(ds, row="time", col="model", height=4, aspect=1.2)

# 3. Map SpatialImshowPlot to each facet using map_monet
grid.map_monet(
    SpatialImshowPlot,
    cmap="viridis",
    coastlines=True,
    add_colorbar=True,
    label="Temperature (C)",
)

# 4. Final adjustments
plt.tight_layout()

# 5. Save the figure
grid.save("facet_grid_spatial_plot.png", dpi=300)

print("Generated 'facet_grid_spatial_plot.png'")
