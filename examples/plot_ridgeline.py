"""
Ridgeline Plot
==============

This example demonstrates how to create a Ridgeline plot (joyplot).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from monet_plots.plots.ridgeline import RidgelinePlot

# 1. Prepare sample data
# Grouped data for different categories
np.random.seed(42)
n_points = 200
categories = ['Group A', 'Group B', 'Group C', 'Group D']
data = []

for i, cat in enumerate(categories):
    values = np.random.normal(i * 2, 1.0, n_points)
    df = pd.DataFrame({'measurement': values, 'group': cat})
    data.append(df)

df_all = pd.concat(data)

# 2. Initialize and create the plot
# RidgelinePlot(data, group_dim, x=None, ...)
plot = RidgelinePlot(df_all, 'group', x='measurement', figsize=(10, 8))
plot.plot(
    title="Ridgeline Plot of Different Groups",
    cmap='viridis'
)

plt.show()
