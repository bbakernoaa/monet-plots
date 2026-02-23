"""
Taylor Diagram
==============

**What it's for:**
The Taylor Diagram provides a concise statistical summary of how well a model (or multiple
models) matches a reference dataset (usually observations). It summarizes three
statistics in a single point: the correlation coefficient, the root-mean-square (RMS)
error, and the standard deviation.

**When to use:**
Use this diagram to compare multiple models or different versions of the same model
against a single set of observations. It is a standard tool in climate and meteorological
model evaluation for assessing spatial or temporal patterns.

**How to read:**
*   **Radial Distance from Origin:** Represents the Standard Deviation of the data.
*   **Angular Coordinate (Arc):** Represents the Pearson Correlation Coefficient (R).
*   **Distance from the Reference Point (on the X-axis):** Represents the Centered
    Root-Mean-Square (RMS) error.
*   **Interpretation:** A perfect model would be represented by the "Reference" point
    on the x-axis (Correlation = 1, same Standard Deviation as observations, RMS error = 0).
    The closer a model's point is to the reference point, the better it matches the observations.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from monet_plots.plots.taylor_diagram import TaylorDiagramPlot

# 1. Prepare model data
np.random.seed(42)
n_points = 1000
obs = np.random.normal(0, 1.2, n_points)
model1 = 0.9 * obs + np.random.normal(0, 0.5, n_points)
model2 = 0.7 * obs + np.random.normal(0, 0.8, n_points)

df = pd.DataFrame({"obs": obs, "Model A": model1, "Model B": model2})

# 2. Initialize and create the plot
# TaylorDiagramPlot calculates statistics from the DataFrame
plot = TaylorDiagramPlot(df, col1="obs", col2=["Model A", "Model B"], scale=1.5)
plot.plot()

plt.title("Model Comparison Taylor Diagram")
plt.show()
