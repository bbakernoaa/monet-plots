"""
ROC Curve
=========

**What it's for:**
The Receiver Operating Characteristic (ROC) curve evaluates the discrimination ability of a
forecast system—how well it can distinguish between an event occurring and not occurring.
It is independent of bias and calibration.

**When to use:**
Use this curve to compare the overall skill of different models or to determine the
optimal threshold for a binary classifier. It is a standard tool in both signal detection
theory and meteorological verification.

**How to read:**
*   **X-axis:** Probability of False Detection (POFD) or False Alarm Rate.
*   **Y-axis:** Probability of Detection (POD) or Hit Rate.
*   **The Curve:** Each point on the curve represents a different decision threshold.
*   **Top-Left Corner:** Represents a perfect forecast (POD = 1, POFD = 0).
*   **Diagonal Line:** Represents a "no-skill" forecast, equivalent to random guessing.
*   **Area Under the Curve (AUC):** A value of 1.0 indicates perfect discrimination, while
    0.5 indicates no skill. The higher the AUC, the better the model's potential skill.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from monet_plots.plots.roc_curve import ROCCurvePlot

# 1. Prepare sample data
np.random.seed(42)  # for reproducibility

# Simulate POD and POFD values for different thresholds
# A good model will have high POD for low POFD
thresholds = np.linspace(0, 1, 50)
pod = 0.5 * (1 + np.tanh(5 * (thresholds - 0.5))) + np.random.normal(0, 0.05, 50)
pofd = thresholds + np.random.normal(0, 0.03, 50)

# Ensure values are within [0, 1] and sorted for plotting
pod = np.clip(pod, 0, 1)
pofd = np.clip(pofd, 0, 1)

# Sort by POFD to ensure correct curve plotting
df = (
    pd.DataFrame({"pofd": pofd, "pod": pod})
    .sort_values(by="pofd")
    .reset_index(drop=True)
)

# 2. Initialize and create the plot
plot = ROCCurvePlot(figsize=(8, 8))
plot.plot(df, x_col="pofd", y_col="pod", color="blue", linewidth=2)

# 3. Add titles and labels
plot.ax.set_title("Receiver Operating Characteristic (ROC) Curve")
# Legend includes AUC automatically if show_auc=True (default)

plt.tight_layout()
plt.show()
