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

from monet_plots.plots.scatter import Scatter
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
plot = Scatter(ax=ax, data=df)
plot.plot(
    x="observation",
    y="model",
    xlabel=f"Observed Temperature ({ds['temperature'].attrs['units']})",
    ylabel=f"Modeled Temperature ({ds['temperature'].attrs['units']})",
    title="Model vs. Observation Scatter Plot",
)

# 3. Add a 1-to-1 line for reference
plot.add_one_to_one()

# 4. Add a linear regression line
plot.add_regression()

plt.show()
