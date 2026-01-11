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

from monet_plots.plots.roc_curve import ROCCurvePlot

# 1. Create synthetic probabilistic forecast data
np.random.seed(42)
n_samples = 1000
# Forecast probabilities for an event (e.g., gust > 40 mph)
forecast_prob = np.random.rand(n_samples)
# Corresponding binary observations (e.g., did the gust occur?)
observed_binary = (forecast_prob > np.random.rand(n_samples)).astype(int)


# 2. Calculate POD and POFD for a range of thresholds
thresholds = np.linspace(0, 1, 21)
roc_data = []
for thresh in thresholds:
    forecast_binary = (forecast_prob >= thresh).astype(int)
    hits = np.sum((forecast_binary == 1) & (observed_binary == 1))
    misses = np.sum((forecast_binary == 0) & (observed_binary == 1))
    false_alarms = np.sum((forecast_binary == 1) & (observed_binary == 0))
    correct_negatives = np.sum((forecast_binary == 0) & (observed_binary == 0))
    pod = hits / (hits + misses) if (hits + misses) > 0 else 0
    pofd = false_alarms / (false_alarms + correct_negatives) if (false_alarms + correct_negatives) > 0 else 0
    roc_data.append({"threshold": thresh, "pod": pod, "pofd": pofd})

roc_df = pd.DataFrame(roc_data)

# 3. Create and display the plot
fig, ax = plt.subplots(figsize=(8, 8))
plot = ROCCurvePlot(ax=ax, fig=fig)
plot.plot(data=roc_df, x_col="pofd", y_col="pod")
ax.set_title("ROC Curve for a Synthetic Forecast")
plt.show()
