"""
Profile Plot
============

**What it's for:**
A Profile plot visualizes the vertical distribution of a variable (e.g., temperature,
humidity, or pollutant concentration) through the atmosphere or a body of water.

**When to use:**
Use this for atmospheric soundings (radiosondes), LIDAR/SODAR measurements, or model
vertical grid evaluation. It is essential for understanding the stability of the
atmosphere or the structure of the planetary boundary layer.

**How to read:**
*   **X-axis:** The variable of interest (e.g., Temperature, Mixing Ratio).
*   **Y-axis:** Altitude (meters, kilometers) or Pressure (hPa). In meteorology, it is common
    to use pressure as a vertical coordinate.
*   **Interpretation:** The slope of the line indicates the vertical gradient of the
    variable (e.g., the lapse rate for temperature).
"""

import matplotlib.pyplot as plt
import numpy as np

from monet_plots.plots.profile import ProfilePlot

# 1. Prepare sample 1D data
altitude = np.linspace(0, 10000, 100)  # meters
temperature = 20 - 0.0065 * altitude + 5 * np.sin(altitude / 1000)  # degrees Celsius

# 2. Initialize and create the plot
plot = ProfilePlot(x=temperature, y=altitude, figsize=(7, 9))
plot.plot(color="red", linewidth=2, label="Temperature Profile")

# 3. Add titles and labels
plot.ax.set_title("Atmospheric Temperature Profile")
plot.ax.set_xlabel("Temperature (°C)")
plot.ax.set_ylabel("Altitude (m)")
plot.ax.legend()
plot.ax.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()
