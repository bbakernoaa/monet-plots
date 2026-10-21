"""
Performance Diagram
===================

**What it's for:**
The Performance Diagram is a powerful visualization tool for evaluating categorical (yes/no)
forecasts. It provides a concise summary of several key verification metrics in a single plot,
specifically relating the Success Ratio (1 - False Alarm Ratio) and the Probability of Detection (POD).

**When to use:**
Use this diagram when evaluating the performance of deterministic or thresholded probabilistic
forecasts, especially for rare events where standard accuracy can be misleading. It is
commonly used in meteorology to assess weather warning systems and model skill.

**How to read:**
*   **X-axis:** Success Ratio (the fraction of "yes" forecasts that were correct).
*   **Y-axis:** Probability of Detection (the fraction of "yes" events that were correctly forecast).
*   **Dashed Lines:** Represent Frequency Bias. A bias of 1 (the diagonal) indicates the system
    forecast the event as often as it occurred.
*   **Solid Contours:** Represent the Critical Success Index (CSI). Higher CSI values (moving
    toward the top-right) indicate better overall skill.
*   **Interpretation:** A perfect forecast resides at the top-right corner (Success Ratio = 1, POD = 1).
"""

import matplotlib.pyplot as plt
import pandas as pd

from monet_plots.plots.performance_diagram import PerformanceDiagramPlot

# 1. Prepare sample data with pre-calculated metrics
# Each row represents a different forecast threshold or scenario
data = {
    "success_ratio": [0.2, 0.4, 0.6, 0.8, 0.9],
    "pod": [0.3, 0.6, 0.7, 0.85, 0.92],
    "model": ["Model A"] * 5,  # For a single model
}
df = pd.DataFrame(data)

# 2. Initialize and create the plot
plot = PerformanceDiagramPlot(figsize=(8, 8))
plot.plot(
    df,
    x_col="success_ratio",
    y_col="pod",
    markersize=8,
    color="blue",
    label="Forecast System",
)

# 3. Add titles and labels (optional, but good practice)
plot.ax.set_title("Performance Diagram for a Forecast System")
plot.ax.legend(loc="lower right")  # Add legend for the plotted points

plt.tight_layout()
plt.show()
