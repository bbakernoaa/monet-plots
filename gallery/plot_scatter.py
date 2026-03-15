"""
Scatter Plot with Regression
============================

**What it's for:**
A Scatter Plot visualizes the relationship between two continuous variables. This
implementation often includes a linear regression line to summarize the trend.

**When to use:**
Use this to assess the correlation between model output and observations, or to
explore the relationship between two different physical variables (e.g., temperature
vs. ozone concentration).

**How to read:**
*   **X-axis:** Typically the predictor or independent variable (e.g., Observations).
*   **Y-axis:** Typically the response or dependent variable (e.g., Model).
*   **Regression Line:** Shows the best-fit linear relationship. The slope and
    intercept provide information about systematic bias and scaling.
*   **Interpretation:** The tightness of the point cluster around the line indicates
     the strength of the correlation. Points on the 1:1 diagonal would represent
     a perfect match.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from monet_plots.plots.scatter import ScatterPlot

# 1. Prepare sample data
np.random.seed(42)  # for reproducibility
n_samples = 100

# Simulate two correlated variables
x_data = np.random.normal(loc=10, scale=2, size=n_samples)
y_data = 0.7 * x_data + np.random.normal(loc=5, scale=1.5, size=n_samples)

df = pd.DataFrame({"predictor": x_data, "response": y_data})

# 2. Initialize and create the plot
plot = ScatterPlot(
    df=df,
    x="predictor",
    y="response",
    title="Scatter Plot of Response vs. Predictor",
    figsize=(9, 7),
)
plot.plot(
    scatter_kws={"alpha": 0.7, "s": 60, "edgecolor": "w"},  # kwargs for scatter points
    line_kws={"color": "red", "linewidth": 2},  # kwargs for regression line
)

# 3. Add titles and labels
plot.ax.set_xlabel("Predictor Variable")
plot.ax.set_ylabel("Response Variable")
plot.ax.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()
