"""
Vertical Slice Plot
===================

**What it's for:**
The `VerticalSlice` class visualizes a 2D cross-section of the atmosphere,
typically showing how a variable changes with both altitude and distance (or time).

**When to use:**
Use this to examine the vertical structure of features like the planetary
boundary layer, fronts, or pollutant plumes along a specific path or at a
cross-section of a model domain.

**How to read:**
*   **X-axis:** Distance or Horizontal Location.
*   **Y-axis:** Altitude or Pressure.
*   **Colors/Contours:** Represent the magnitude of the variable across the slice.
"""

import matplotlib.pyplot as plt
import numpy as np

from monet_plots.plots.profile import VerticalSlice

# 1. Create dummy 2D data for a cross-section
distance = np.linspace(0, 500, 50)
altitude = np.linspace(0, 10, 30)
dist_grid, alt_grid = np.meshgrid(distance, altitude)

# Simulate a pollutant plume rising and moving
z_data = 50 * np.exp(-((dist_grid - 200) ** 2 / 5000 + (alt_grid - 2) ** 2 / 2))

# 2. Initialize and plot
plot = VerticalSlice(x=dist_grid, y=alt_grid, z=z_data, figsize=(10, 6))
plot.plot(cmap="YlOrRd")

# 3. Add titles and labels
plot.ax.set_title("Vertical Cross-Section of Pollutant Plume")
plot.ax.set_xlabel("Distance (km)")
plot.ax.set_ylabel("Altitude (km)")
plot.add_colorbar(label="Concentration (ppb)")

plt.show()
