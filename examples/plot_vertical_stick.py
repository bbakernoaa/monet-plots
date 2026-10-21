"""
Vertical Stick Plot
===================

**What it's for:**
The `StickPlot` class visualizes vertical wind profiles using vectors (sticks)
to represent the magnitude and direction of the wind at different heights.

**When to use:**
Use this to analyze the vertical shear of the wind or the evolution of wind
profiles at a single location over time (if the x-axis represents time).

**How to read:**
*   **Y-axis:** Altitude or Pressure.
*   **Vector Length:** Proportional to the wind speed.
*   **Vector Direction:** Indicates the direction the wind is blowing towards.
"""

import matplotlib.pyplot as plt
import numpy as np

from monet_plots.plots.profile import StickPlot

# 1. Create dummy vertical wind data
altitude = np.linspace(0, 5000, 20)
# U (east-west) and V (north-south) components
u = 5 + 2 * altitude / 1000
v = 2 * np.sin(altitude / 1000)

# 2. Initialize and plot
plot = StickPlot(u=u, v=v, y=altitude, figsize=(6, 8))
# quiver arguments can be passed to plot()
plot.plot(color="blue", scale=50)

# 3. Add titles and labels
plot.ax.set_title("Vertical Wind Stick Plot")
plot.ax.set_xlabel("Horizontal Offset (for visualization)")
plot.ax.set_ylabel("Altitude (m)")
plot.ax.set_xlim(-5, 5)  # Center the sticks

plt.show()
