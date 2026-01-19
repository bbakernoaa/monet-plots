"""
Kde
===

This example demonstrates how to create a Kde.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from monet_plots.plots.kde import KDEPlot

# 1. Prepare sample data
np.random.seed(42) # for reproducibility
mean = [0, 0]
cov = [[1, 0.8], [0.8, 1]] # Correlated variables
data = np.random.multivariate_normal(mean, cov, 1000)
df = pd.DataFrame(data, columns=['Variable A', 'Variable B'])

# 2. Initialize and create the plot
plot = KDEPlot(df=df, x='Variable A', y='Variable B', title="2D Kernel Density Estimate of Correlated Variables", figsize=(8, 7))
plot.plot() # Default KDE plot

# 3. Add titles and labels
plot.ax.set_xlabel("Variable A")
plot.ax.set_ylabel("Variable B")

plt.tight_layout()
plt.show()
