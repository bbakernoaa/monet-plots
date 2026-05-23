"""
Vertical Profile Plot
=====================

**What it's for:**
The `VerticalProfilePlot` class visualizes vertical distributions of a variable,
comparing observations and one or more models across multiple height or
pressure levels.

**When to use:**
Use this to evaluate model performance throughout the atmospheric column.
It supports binned statistics (median, interquartile range) and can represent
data using either shading or box-and-whisker plots.

**How to read:**
*   **X-axis:** The variable of interest (e.g., Temperature, Concentration).
*   **Y-axis:** Altitude or Pressure levels.
*   **Shading/Boxes:** Represent the variability or uncertainty within each
    vertical bin.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from monet_plots.plots.profile import VerticalProfilePlot

# 1. Create dummy data for observations and two models
n_samples = 1000
altitude = np.random.uniform(0, 5000, n_samples)
obs = 100 - 0.01 * altitude + np.random.normal(0, 5, n_samples)
mod1 = obs + np.random.normal(0, 3, n_samples) + 2
mod2 = obs + np.random.normal(0, 10, n_samples) - 5

df = pd.DataFrame(
    {
        "alt": altitude,
        "obs": obs,
        "model_A": mod1,
        "model_B": mod2,
    }
)

# 2. Initialize and plot with shading style
plot = VerticalProfilePlot(
    df,
    obs_col="obs",
    mod_cols=["model_A", "model_B"],
    alt_col="alt",
    bins=15,
    interquartile_style="shading",
    figsize=(8, 10),
)
plot.plot()

# 3. Add titles and labels
plot.ax.set_title("Vertical Profile Comparison (Shading)")
plot.ax.set_xlabel("Concentration (unit)")
plot.ax.set_ylabel("Altitude (m)")

plt.show()
