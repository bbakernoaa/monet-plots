"""
Vertical Box Plot
=================

**What it's for:**
The `VerticalBoxPlot` class visualizes the distribution of a variable within
different altitude or pressure bins using box-and-whisker plots.

**When to use:**
Use this to see the statistical distribution (median, quartiles, outliers) of
a variable at different heights, providing a more detailed view than a
simple mean or median profile.

**How to read:**
*   **Box:** Represents the interquartile range (IQR, 25th to 75th percentile).
*   **Line in Box:** Represents the median.
*   **Whiskers:** Represent the extent of the data beyond the IQR.
*   **Y-axis:** Shows the altitude or pressure bins.
"""

import matplotlib.pyplot as plt
import numpy as np

from monet_plots.plots.profile import VerticalBoxPlot

# 1. Create dummy data
n_samples = 1000
altitude = np.random.uniform(0, 10000, n_samples)
data = 50 + 20 * np.sin(altitude / 2000) + np.random.normal(0, 10, n_samples)

# Define bin thresholds for altitude
thresholds = [0, 2000, 4000, 6000, 8000, 10000]

# 2. Initialize and plot
plot = VerticalBoxPlot(data=data, y=altitude, thresholds=thresholds, figsize=(8, 10))
plot.plot(patch_artist=True, boxprops=dict(facecolor="skyblue"))

# 3. Add titles and labels
plot.ax.set_title("Distribution of Values by Altitude Bin")
plot.ax.set_xlabel("Value")
plot.ax.set_ylabel("Altitude (m)")

plt.show()
