"""
Conditional Bias Plot
=====================

This example demonstrates the Conditional Bias plot, which visualizes how the
model bias (Model - Observation) changes depending on the value of the
observation itself. It helps to identify systematic errors in a model.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from monet_plots.plots.conditional_bias import ConditionalBias

# 1. Create synthetic data with a conditional bias
np.random.seed(42)
n_samples = 1000
# "Observed" data, e.g., temperature
obs_data = np.random.randn(n_samples) * 15 + 288
# "Modeled" data with a bias that depends on the observation value
# Bias = 5% of the observation value minus a constant offset
bias = 0.05 * obs_data - 14.0 + np.random.randn(n_samples) * 2
mod_data = obs_data + bias

df = pd.DataFrame({"observation": obs_data, "model": mod_data})

# 2. Create and display the plot
fig, ax = plt.subplots(figsize=(8, 6))
plot = ConditionalBias(ax=ax, data=df)
plot.plot(
    obs_col="observation",
    mod_col="model",
    nbins=20,
    title="Conditional Bias of Modeled Temperature",
)
ax.set_xlabel("Observed Temperature (K)")
ax.set_ylabel("Model Bias (K)")
plt.show()
