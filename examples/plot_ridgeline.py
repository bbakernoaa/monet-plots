"""
Ridgeline Plot
==============

**What it's for:**
A Ridgeline plot (also known as a Joyplot) visualizes the distribution of a continuous
variable across multiple groups by using partially overlapping density plots.

**When to use:**
Use this to compare how a distribution changes over time (e.g., daily profiles for each
month) or across different experimental groups. It is more space-efficient and often
easier to read than multiple overlapping histograms or KDE plots.

**How to read:**
*   **X-axis:** The value of the variable being measured.
*   **Y-axis:** Represents different categories or groups. Each group has its own
    Kernel Density Estimate (KDE).
*   **Interpretation:** Look for shifts in the peak (mean/mode) or the width (spread)
    of the distributions as you move from one group to another.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from monet_plots.plots.ridgeline import RidgelinePlot

# 1. Prepare sample data
# Grouped data for different categories
np.random.seed(42)
n_points = 200
categories = ["Group A", "Group B", "Group C", "Group D"]
data = []

for i, cat in enumerate(categories):
    values = np.random.normal(i * 2, 1.0, n_points)
    df = pd.DataFrame({"measurement": values, "group": cat})
    data.append(df)

df_all = pd.concat(data)

# 2. Initialize and create the plot
# RidgelinePlot(data, group_dim, x=None, ...)
plot = RidgelinePlot(df_all, "group", x="measurement", figsize=(10, 8))
plot.plot(title="Ridgeline Plot of Different Groups", cmap="viridis")

plt.show()
