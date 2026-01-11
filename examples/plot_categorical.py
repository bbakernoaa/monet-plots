"""
Categorical Plot
================

This example demonstrates the Categorical Plot, used to compare the frequency
of different categories between two datasets (e.g., model vs. observation).
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from monet_plots.plots.categorical import Categorical

# 1. Create synthetic categorical data
np.random.seed(42)
categories = ["Clear", "Cloudy", "Rainy", "Stormy"]
n_samples = 1000
# "Observed" categories
obs_data = np.random.choice(categories, n_samples, p=[0.4, 0.3, 0.2, 0.1])
# "Modeled" categories - slightly different probabilities
mod_data = np.random.choice(categories, n_samples, p=[0.35, 0.35, 0.2, 0.1])

df = pd.DataFrame({"observed": obs_data, "modeled": mod_data})

# 2. Create and display the plot
fig, ax = plt.subplots(figsize=(8, 6))
plot = Categorical(ax=ax, data=df)
plot.plot(
    obs_col="observed",
    mod_col="modeled",
    title="Comparison of Observed vs. Modeled Weather Categories",
)
plt.show()
