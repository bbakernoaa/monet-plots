"""
Wind Rose
=========

**What it's for:**
A Wind Rose is a graphical tool used by meteorologists to give a succinct view of how wind
speed and direction are typically distributed at a particular location.

**When to use:**
Use this to characterize the wind climatology of a site. It is critical for siting
wind turbines, designing airport runways, and understanding the dispersion of
pollutants from a local source.

**How to read:**
*   **Direction (Circular/Compass):** The "petals" point toward the direction from which
    the wind is blowing.
*   **Length of Petals:** Indicates the frequency (percentage of time) the wind blows
    from that direction.
*   **Color within Petals:** Represents different wind speed categories (e.g., 0-2 m/s,
    2-5 m/s, etc.).
*   **Interpretation:** A long petal pointing North indicates that the wind frequently
    blows from the North. The distribution of colors within that petal shows the
    distribution of wind speeds for that specific direction.
"""

import matplotlib.pyplot as plt
import numpy as np

from monet_plots.plots.windrose import Windrose

# 1. Prepare sample data
np.random.seed(42)
n_samples = 1000
ws = np.random.gamma(2, 2, n_samples)  # Wind speed
wd = np.random.uniform(0, 360, n_samples)  # Wind direction

# 2. Initialize and create the plot
plot = Windrose(wd=wd, ws=ws, figsize=(8, 8))
plot.plot()

plt.title("Sample Windrose")
plt.show()
