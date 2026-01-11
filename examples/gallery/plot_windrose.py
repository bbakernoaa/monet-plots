"""
Wind Rose Plot
==============

This example demonstrates how to create a Wind Rose plot using the
:class:`monet_plots.plots.windrose.WindRose` class.

A Wind Rose provides a compact and intuitive view of how wind speed and
direction are distributed at a particular location.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from monet_plots.plots.windrose import Windrose

# 1. Create synthetic wind data for a single point over a long time series
np.random.seed(42)
n_samples = 2000
# Wind direction (degrees): Prevailing westerly winds with some variability
wind_direction = np.random.randn(n_samples) * 45 + 270
wind_direction = wind_direction % 360  # Wrap around 360 degrees

# Wind speed (m/s)
wind_speed = np.random.power(2, n_samples) * 20

# 2. Create and display the Wind Rose
# The plot is created on a polar projection.
fig = plt.figure(figsize=(8, 8))
plot = Windrose(wd=wind_direction, ws=wind_speed, fig=fig)
plot.plot(
    bins=np.arange(0, 361, 45),
    rose_bins=[0, 2, 5, 10, 15, 20],
)
fig.suptitle("Wind Rose for a Single Location")

plt.show()
