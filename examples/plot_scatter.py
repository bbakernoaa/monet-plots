"""
Scatter
=======

This example demonstrates how to create a Scatter.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from monet_plots.plots.scatter import ScatterPlot

# 1. Prepare sample data
np.random.seed(42) # for reproducibility
n_samples = 100

# Simulate two correlated variables
x_data = np.random.normal(loc=10, scale=2, size=n_samples)
y_data = 0.7 * x_data + np.random.normal(loc=5, scale=1.5, size=n_samples)

df = pd.DataFrame({
    'predictor': x_data,
    'response': y_data
})

# 2. Initialize and create the plot
plot = ScatterPlot(df=df, x='predictor', y='response', title="Scatter Plot of Response vs. Predictor", figsize=(9, 7))
plot.plot(
    scatter_kws={'alpha': 0.7, 's': 60, 'edgecolor': 'w'}, # kwargs for scatter points
    line_kws={'color': 'red', 'linewidth': 2} # kwargs for regression line
)

# 3. Add titles and labels
plot.ax.set_xlabel("Predictor Variable")
plot.ax.set_ylabel("Response Variable")
plot.ax.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
