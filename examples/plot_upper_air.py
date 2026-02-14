"""
Upper Air Plot
==============

**What it's for:**
An Upper Air plot is a standard meteorological visualization that combines
geopotential height (represented as contours) and wind (represented as barbs)
on a single map for a specific atmospheric pressure level.

**When to use:**
Use this to analyze the synoptic-scale weather patterns at various levels of the
atmosphere (e.g., 850, 700, 500, or 250 hPa). It is essential for identifying
ridges, troughs, jet streams, and other features that drive surface weather.

**How to read:**
*   **Contours:** Lines of equal geopotential height (similar to isobars on a
    surface map).
*   **Wind Barbs:** Indicate the wind direction and speed at specific points.
*   **Interpretation:** The relationship between the height contours and the wind
    barbs indicates the atmospheric flow. For example, in the mid-latitudes, the
    wind typically flows roughly parallel to the height contours (geostrophic flow).
"""

import matplotlib.pyplot as plt
import numpy as np

from monet_plots.plots.upper_air import UpperAir

# 1. Prepare sample data
lats = np.linspace(30, 50, 20)
lons = np.linspace(-125, -70, 30)
hgt = np.random.uniform(5000, 6000, (20, 30))
u = np.random.uniform(-20, 20, (20, 30))
v = np.random.uniform(-20, 20, (20, 30))

# 2. Initialize and plot
plot = UpperAir(lat=lats, lon=lons, hgt=hgt, u=u, v=v, figsize=(10, 8))
plot.plot()

plt.title("Upper Air Example (500 hPa)")
plt.show()
