"""
Spatial Bias Scatter
====================

This example demonstrates how to create a Spatial Bias Scatter plot.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from monet_plots.plots.spatial_bias_scatter import SpatialBiasScatterPlot

# 1. Prepare sample data
n_points = 100
df = pd.DataFrame({
    'latitude': np.random.uniform(30, 50, n_points),
    'longitude': np.random.uniform(-125, -70, n_points),
    'obs': np.random.uniform(0, 50, n_points),
    'model': np.random.uniform(0, 50, n_points)
})

# 2. Initialize and plot
# Bias is calculated as col2 - col1
plot = SpatialBiasScatterPlot(df, col1='obs', col2='model', projection=ccrs.PlateCarree(), figsize=(10, 8))
plot.plot()

plt.title("Spatial Bias Scatter Example")
plt.show()
