"""
Spatial Plot
============

**What it's for:**
The `SpatialPlot` class is a high-level wrapper for creating geographic maps with
pre-configured styles, projections, and features (like coastlines and states).

**When to use:**
Use this for any plot that requires a map background. It serves as the base for
more specialized spatial plots (like contours or scatter plots) but can be used
standalone for custom map visualizations.

**How to read:**
*   **Axes:** Represents geographic space using a specific map projection (e.g.,
    Plate Carrée or Lambert Conformal).
*   **Colorbar:** Shows the relationship between colors on the map and the data values.
*   **Interpretation:** Allows for the visualization of the spatial extent and
    gradients of a variable across a region.
"""

import numpy as np
import matplotlib.pyplot as plt
from monet_plots.plots.spatial import SpatialPlot

data = np.random.random((20, 30)) * 100
plot = SpatialPlot(figsize=(10, 8))
im = plot.ax.pcolormesh(data, cmap="viridis", shading="auto")
plot.ax.set_title("Basic Spatial Plot")
plot.add_colorbar(im, label="Value")
plt.show()
