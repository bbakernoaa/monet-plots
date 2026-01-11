"""
Scatter Plot
============

This example demonstrates how to create a scatter plot, including a 1-to-1
line and a linear regression fit, using the
:class:`monet_plots.plots.scatter.Scatter` class.

This type of plot is very common for comparing a modeled variable against
observations.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from monet_plots.plots.scatter import ScatterPlot
from data import create_dataset

# 1. Create synthetic data for model vs. observation comparison
ds = create_dataset()
# Use surface temperature as our "observation"
obs = ds["temperature"].isel(time=0, level=0).values.flatten()
# Create a "model" by adding some noise and a slight bias
mod = obs + np.random.randn(len(obs)) * 2 + 1.5
df = pd.DataFrame({'observation': obs, 'model': mod})


# 2. Create and display the scatter plot
fig, ax = plt.subplots(figsize=(8, 8))
plot = ScatterPlot(
    df=df,
    x="observation",
    y="model",
    title="Model vs. Observation Scatter Plot",
    ax=ax,
    fig=fig,
)
plot.plot()
ax.set_xlabel(f"Observed Temperature ({ds['temperature'].attrs['units']})")
ax.set_ylabel(f"Modeled Temperature ({ds['temperature'].attrs['units']})")

# 3. Add a 1-to-1 line for reference
x_lim = ax.get_xlim()
y_lim = ax.get_ylim()
ax.plot([min(x_lim[0], y_lim[0]), max(x_lim[1], y_lim[1])],
        [min(x_lim[0], y_lim[0]), max(x_lim[1], y_lim[1])],
        color='black', linestyle='--')

plt.show()
