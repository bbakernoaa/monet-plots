"""
Sp Scatter Bias
===============

This example demonstrates how to create a Sp Scatter Bias.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from monet_plots.plots.sp_scatter_bias import SpScatterBiasPlot

# 1. Prepare sample data
np.random.seed(42) # for reproducibility
n_points = 500

# Simulate random latitude and longitude points
lat = np.random.uniform(20, 50, n_points)
lon = np.random.uniform(-120, -70, n_points)

# Simulate reference and comparison values
reference_values = 10 + 5 * np.random.rand(n_points)
# Introduce a spatial bias: higher values in the west, lower in the east
comparison_values = reference_values + (lon / 100 + np.random.normal(0, 0.5, n_points))

df = pd.DataFrame({
    'latitude': lat,
    'longitude': lon,
    'reference_value': reference_values,
    'comparison_value': comparison_values
})

# 2. Initialize and create the plot
plot = SpScatterBiasPlot(
    df=df,
    col1='reference_value',
    col2='comparison_value',
    figsize=(10, 8),
    projection=ccrs.PlateCarree(),
    extent=[-125, -65, 15, 55],
    coastlines=True,
    countries=True,
)
plot.plot(
    cmap='RdBu_r', # Red-Blue colormap, reversed for positive=red, negative=blue
    edgecolor='black',
    linewidth=0.5,
    alpha=0.8
)

# 3. Add titles and labels
plot.ax.set_title("Spatial Bias Plot: Comparison vs. Reference")
# Map elements (coastlines, borders) are added by default via draw_map

plt.tight_layout()
plt.show()
