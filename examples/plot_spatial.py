"""
Spatial Plot
============

This example demonstrates how to create a basic spatial plot.
"""

import numpy as np
import matplotlib.pyplot as plt
from monet_plots.plots.spatial import SpatialPlot

data = np.random.random((20, 30)) * 100
plot = SpatialPlot(figsize=(10, 8))
im = plot.ax.pcolormesh(data, cmap='viridis', shading='auto')
plot.ax.set_title("Basic Spatial Plot")
plt.colorbar(im, ax=plot.ax, label='Value')
plt.show()
