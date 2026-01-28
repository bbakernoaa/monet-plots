"""
Conditional Quantile
====================

**What it's for:**
A Conditional Quantile plot provides a comprehensive view of model performance by showing
the distribution of modeled values conditioned on the observed values. It goes beyond simple
mean bias by showing the spread and variability of the model across the entire data range.

**When to use:**
Use this when you need a detailed diagnosis of model error distributions. It is
particularly useful for understanding how the model handles extremes and whether the
uncertainty in the model increases or decreases with the magnitude of the variable.

**How to read:**
*   **X-axis:** The observed value, divided into bins.
*   **Y-axis:** The modeled value.
*   **1:1 Line:** Represents a perfect match between model and observations.
*   **Central Line (Median):** Shows the 50th percentile of the model for each observation bin.
*   **Shaded Regions/Quantiles:** Represent different percentiles (e.g., 25th-75th, 5th-95th).
    A narrow band indicates consistent model behavior, while a wide band indicates high
    variability/uncertainty for that observed value.
"""

import pandas as pd
import numpy as np
from monet_plots.plots.conditional_quantile import ConditionalQuantilePlot

# Create dummy model vs obs
obs = np.random.exponential(10, 1000)
mod = obs * 0.8 + np.random.normal(0, 5, 1000) + 2

df = pd.DataFrame({"obs": obs, "mod": mod})

# Initialize and plot
plot = ConditionalQuantilePlot(df, obs_col="obs", mod_col="mod", bins=10)
plot.plot(show_points=True)
plot.save("conditional_quantile_example.png")
print("Conditional quantile plot saved to conditional_quantile_example.png")
