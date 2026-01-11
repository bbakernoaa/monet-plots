"""
KDE Plot
========

This example demonstrates the Kernel Density Estimate (KDE) plot, used for
visualizing the probability density of a dataset. It can be thought of as a
smoothed histogram.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from monet_plots.plots.kde import KDE

# 1. Create synthetic data
np.random.seed(42)
n_samples = 1000
# Data from two different distributions
data1 = np.random.normal(0, 1, n_samples)
data2 = np.random.normal(3, 1.5, n_samples)
df = pd.DataFrame({"distribution_A": data1, "distribution_B": data2})

# 2. Create and display the plot
fig, ax = plt.subplots(figsize=(8, 6))
plot = KDE(ax=ax, data=df)
plot.plot(
    columns=["distribution_A", "distribution_B"],
    title="Kernel Density Estimate of Two Distributions",
)
ax.set_xlabel("Value")
ax.legend()
plt.show()
