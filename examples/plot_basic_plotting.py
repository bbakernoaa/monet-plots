"""
Basic Plotting
==============

This example demonstrates how to create a Basic Plotting.
"""

import numpy as np
from monet_plots import SpatialPlot

# Step 1: Prepare spatial data
# Create a 2D array representing spatial data
data = np.random.random((20, 30)) * 100

# Step 2: Initialize the plot
plot = SpatialPlot(figsize=(10, 8))

# Step 3: Plot the data
# SpatialPlot sets up the map axes. We use standard matplotlib/cartopy methods to plot.
plot.ax.pcolormesh(data)
plot.ax.set_title("Basic Spatial Plot")

# Step 4: Add labels and save
plot.ax.set_xlabel("Longitude")
plot.ax.set_ylabel("Latitude")
plot.save("basic_spatial.png")

# Step 5: Close the plot
plot.close()

print("Basic spatial plot saved successfully!")
