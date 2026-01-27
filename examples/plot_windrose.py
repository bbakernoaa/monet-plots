"""
Windrose
========

This example demonstrates how to create a Windrose plot.
"""

import numpy as np
import matplotlib.pyplot as plt
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
