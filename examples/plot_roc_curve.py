"""
ROC Curve
=========

This example demonstrates the Receiver Operating Characteristic (ROC) Curve,
which is used to evaluate the performance of a binary classifier or a
probabilistic forecast for a specific event.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from monet_plots.plots.roc_curve import ROCCurve

# 1. Create synthetic probabilistic forecast data
np.random.seed(42)
n_samples = 1000
# Forecast probabilities for an event (e.g., gust > 40 mph)
forecast_prob = np.random.rand(n_samples)
# Corresponding binary observations (e.g., did the gust occur?)
observed_binary = (forecast_prob > np.random.rand(n_samples)).astype(int)

df = pd.DataFrame(
    {"forecast_prob": forecast_prob, "observed_binary": observed_binary}
)

# 2. Create and display the plot
fig, ax = plt.subplots(figsize=(8, 8))
plot = ROCCurve(ax=ax, data=df)
plot.plot(
    prob_col="forecast_prob",
    obs_col="observed_binary",
    title="ROC Curve for a Synthetic Forecast",
)
plt.show()
