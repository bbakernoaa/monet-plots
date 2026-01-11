"""
Scorecard Plot
==============

This example demonstrates the Scorecard plot, a compact way to display multiple
performance metrics for different models or forecast configurations.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from monet_plots.plots.scorecard import ScorecardPlot

# 1. Create a synthetic DataFrame of performance metrics in long format
np.random.seed(42)
models = [f"Model_{chr(65+i)}" for i in range(5)]
metrics = ["RMSE", "Bias", "Correlation", "MAE"]
data = []
for model in models:
    for metric in metrics:
        value = np.random.rand()
        if metric == "Bias":
            value = value * 2 - 1  # Center bias around zero
        data.append({"Model": model, "Metric": metric, "Value": value})
df = pd.DataFrame(data)


# 2. Create and display the plot
fig, ax = plt.subplots(figsize=(8, 6))
plot = ScorecardPlot(ax=ax, fig=fig)
plot.plot(data=df, x_col="Metric", y_col="Model", val_col="Value", center=0)
plt.show()
