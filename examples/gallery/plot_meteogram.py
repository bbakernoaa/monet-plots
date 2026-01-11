"""
Meteogram Plot
==============

This example demonstrates the Meteogram plot, which displays multiple
meteorological variables over time for a single location.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from monet_plots.plots.meteogram import Meteogram

# 1. Create synthetic meteogram data
np.random.seed(42)
n_times = 48
times = pd.to_datetime(pd.date_range("2023-01-01", periods=n_times, freq="H"))
df = pd.DataFrame(index=times)
df['temperature'] = np.sin(np.linspace(0, 4 * np.pi, n_times)) * 10 + 280
df['pressure'] = -np.cos(np.linspace(0, 4 * np.pi, n_times)) * 5 + 1010
df['wind_speed'] = np.random.rand(n_times) * 15
df['precipitation'] = np.random.power(0.5, n_times) * 10

# 2. Create and display the plot
plot = Meteogram(data=df)
fig = plot.plot()
fig.suptitle("Meteogram for a Single Location", y=1.0)
plt.show()
