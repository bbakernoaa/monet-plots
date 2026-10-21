"""
Basic Spatial Plotting
======================

**What it's for:**
This example demonstrates the foundational `SpatialPlot` class, which provides a
consistent interface for creating map-based visualizations in MONET Plots.

**When to use:**
Use `SpatialPlot` when you need full control over a map-based visualization and want
to use standard Matplotlib or Cartopy commands on a pre-configured axes that
includes geographic features.

**How to read:**
*   **Axes:** The plot is in geographic coordinates (typically Latitude/Longitude).
*   **Features:** It automatically includes geographic context like coastlines and
    borders to provide spatial orientation.
*   **Interpretation:** Data is plotted directly onto the map; the location of colors
    or markers corresponds to their real-world geographical position.
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
