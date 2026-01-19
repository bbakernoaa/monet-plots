"""
Roc Curve
=========

This example demonstrates how to create a Roc Curve.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from monet_plots.plots.roc_curve import ROCCurvePlot

# 1. Prepare sample data
np.random.seed(42) # for reproducibility

# Simulate POD and POFD values for different thresholds
# A good model will have high POD for low POFD
thresholds = np.linspace(0, 1, 50)
pod = 0.5 * (1 + np.tanh(5 * (thresholds - 0.5))) + np.random.normal(0, 0.05, 50)
pofd = thresholds + np.random.normal(0, 0.03, 50)

# Ensure values are within [0, 1] and sorted for plotting
pod = np.clip(pod, 0, 1)
pofd = np.clip(pofd, 0, 1)

# Sort by POFD to ensure correct curve plotting
df = pd.DataFrame({'pofd': pofd, 'pod': pod}).sort_values(by='pofd').reset_index(drop=True)

# 2. Initialize and create the plot
plot = ROCCurvePlot(figsize=(8, 8))
plot.plot(
    df,
    x_col='pofd',
    y_col='pod',
    color='blue',
    linewidth=2
)

# 3. Add titles and labels
plot.ax.set_title("Receiver Operating Characteristic (ROC) Curve")
# Legend includes AUC automatically if show_auc=True (default)

plt.tight_layout()
plt.show()
