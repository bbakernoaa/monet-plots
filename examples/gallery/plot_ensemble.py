"""
Ensemble Plot
=============

This example demonstrates the Ensemble plot, which is used to visualize the
spread of an ensemble forecast over time for a single variable.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from monet_plots.plots.ensemble import Ensemble

# 1. Create synthetic ensemble forecast data
np.random.seed(42)
n_times = 24
n_members = 10
times = pd.to_datetime(pd.date_range("2023-01-01", periods=n_times, freq="H"))

# Create a base trend
base_trend = np.sin(np.linspace(0, 2 * np.pi, n_times)) * 5 + 280

# Create ensemble members by adding different random walks to the trend
ensemble_data = np.zeros((n_times, n_members))
for i in range(n_members):
    ensemble_data[:, i] = base_trend + np.random.randn(n_times).cumsum() * 0.5

# Create a DataFrame
df = pd.DataFrame(ensemble_data, index=times, columns=[f"member_{i}" for i in range(n_members)])
df.index.name = "time"

# 2. Create and display the plot
fig, ax = plt.subplots(figsize=(10, 6))
plot = Ensemble(ax=ax, data=df)
plot.plot(
    title="Ensemble Forecast for Temperature",
)
ax.set_ylabel("Temperature (K)")
ax.set_xlabel("Time")
plt.show()
