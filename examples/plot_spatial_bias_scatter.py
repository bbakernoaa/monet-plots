"""
Spatial Bias Scatter Plot
=========================

**What it's for:**
The Spatial Bias Scatter plot visualizes model performance (specifically bias) at
discrete point locations, such as air quality monitoring stations, on a geographic map.

**When to use:**
Use this to identify regional patterns in model errors. For example, it can reveal if
a model consistently over-predicts in urban areas vs. rural areas, or if there is a
systematic bias along a coastline.

**How to read:**
*   **Markers:** Each point on the map represents a specific location (e.g., a station).
*   **Marker Color:** Typically represents the Mean Bias (Model - Observation). A
    diverging colormap (e.g., Blue-White-Red) is often used where red indicates
    over-prediction and blue indicates under-prediction.
*   **Interpretation:** Look for geographic clusters of the same color, which
    suggest localized systematic errors in the model.
"""

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from monet_plots.plots.spatial_bias_scatter import SpatialBiasScatterPlot

# 1. Prepare sample data (CONUS region)
n_points = 100
df = pd.DataFrame(
    {
        "latitude": np.random.uniform(30, 50, n_points),
        "longitude": np.random.uniform(-125, -70, n_points),
        "obs": np.random.uniform(0, 50, n_points),
        "model": np.random.uniform(0, 50, n_points),
    }
)

# 2. Initialize and plot with CONUS extent to match the data
# Bias is calculated as col2 - col1
# Set extent to [lon_min, lon_max, lat_min, lat_max] to match the data coverage
plot = SpatialBiasScatterPlot(
    df,
    col1="obs",
    col2="model",
    projection=ccrs.PlateCarree(),
    figsize=(10, 8),
    extent=[-130, -65, 25, 55],  # CONUS extent
    coastlines=True,
    states=True,
)
plot.plot()

plt.title("Spatial Bias Scatter Example")
plt.show()
