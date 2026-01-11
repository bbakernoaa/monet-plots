"""
Brier Decomposition
===================

This example demonstrates the Brier Decomposition plot, which is used to
evaluate the accuracy of probabilistic forecasts.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from monet_plots.plots.brier_decomposition import BrierDecomposition

# 1. Create synthetic probabilistic forecast data
np.random.seed(42)
n_samples = 500
# Forecast probabilities (e.g., probability of rain)
forecast_prob = np.sort(np.random.rand(n_samples))
# Corresponding binary observations (e.g., it actually rained or not)
# Create a dependency on the forecast probability for realism
observed_binary = (forecast_prob > np.random.rand(n_samples) * 0.7).astype(int)

df = pd.DataFrame(
    {"forecast_prob": forecast_prob, "observed_binary": observed_binary}
)

# 2. Create and display the plot
fig, ax = plt.subplots(figsize=(8, 6))
plot = BrierDecomposition(ax=ax, data=df)
plot.plot(
    prob_col="forecast_prob",
    obs_col="observed_binary",
    bins=np.linspace(0, 1, 11),  # 10 bins from 0 to 1
    title="Brier Decomposition of a Synthetic Forecast",
)
plt.show()
