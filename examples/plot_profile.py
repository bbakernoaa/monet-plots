"""
Profile
=======

This example demonstrates how to create a Profile.
"""

import numpy as np
import matplotlib.pyplot as plt
from monet_plots.plots.profile import ProfilePlot

# 1. Prepare sample 1D data
altitude = np.linspace(0, 10000, 100) # meters
temperature = 20 - 0.0065 * altitude + 5 * np.sin(altitude / 1000) # degrees Celsius

# 2. Initialize and create the plot
plot = ProfilePlot(x=temperature, y=altitude, figsize=(7, 9))
plot.plot(color='red', linewidth=2, label='Temperature Profile')

# 3. Add titles and labels
plot.ax.set_title("Atmospheric Temperature Profile")
plot.ax.set_xlabel("Temperature (°C)")
plot.ax.set_ylabel("Altitude (m)")
plot.ax.legend()
plot.ax.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
