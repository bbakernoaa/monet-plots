"""
Reliability Diagram
===================

This example demonstrates the Reliability Diagram, a tool for evaluating the
reliability of probabilistic forecasts.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from monet_plots.plots.reliability_diagram import ReliabilityDiagram

# 1. Create synthetic probabilistic forecast data
np.random.seed(42)
n_samples = 1000
# Forecast probabilities (e.g., probability of precipitation > 1mm)
forecast_prob = np.sort(np.random.rand(n_samples))
# Corresponding binary observations (e.g., did precipitation > 1mm occur?)
# For a reliable forecast, the observed frequency should match the forecast probability
observed_binary = (forecast_prob > np.random.rand(n_samples) * 0.5).astype(int)

df = pd.DataFrame(
    {"forecast_prob": forecast_prob, "observed_binary": observed_binary}
)

# 2. Create and display the plot
fig, ax = plt.subplots(figsize=(8, 8))
plot = ReliabilityDiagram(ax=ax, data=df)
plot.plot(
    prob_col="forecast_prob",
    obs_col="observed_binary",
    bins=np.linspace(0, 1, 11),  # Use 10 bins
    title="Reliability Diagram for a Synthetic Forecast",
)
plt.show()
