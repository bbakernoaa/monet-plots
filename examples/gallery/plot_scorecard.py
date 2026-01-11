"""
Scorecard Plot
==============

This example demonstrates the Scorecard plot, a compact way to display multiple
performance metrics for different models or forecast configurations.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from monet_plots.plots.scorecard import Scorecard

# 1. Create a synthetic DataFrame of performance metrics
np.random.seed(42)
models = [f"Model_{chr(65+i)}" for i in range(5)]
metrics = ["RMSE", "Bias", "Correlation", "MAE"]
data = np.random.rand(len(models), len(metrics))
df = pd.DataFrame(data, index=models, columns=metrics)
# Adjust Bias to be centered around zero
df["Bias"] = df["Bias"] * 2 - 1

# 2. Create and display the plot
fig, ax = plt.subplots(figsize=(8, 6))
plot = Scorecard(ax=ax, data=df)
plot.plot(
    title="Performance Metrics Scorecard",
)
plt.show()
