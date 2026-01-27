"""
Taylor Diagram
==============

This example demonstrates how to create a Taylor Diagram for model comparison.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from monet_plots.plots.taylor_diagram import TaylorDiagramPlot

# 1. Prepare model data
np.random.seed(42)
n_points = 1000
obs = np.random.normal(0, 1.2, n_points)
model1 = 0.9 * obs + np.random.normal(0, 0.5, n_points)
model2 = 0.7 * obs + np.random.normal(0, 0.8, n_points)

df = pd.DataFrame({
    'obs': obs,
    'Model A': model1,
    'Model B': model2
})

# 2. Initialize and create the plot
# TaylorDiagramPlot calculates statistics from the DataFrame
plot = TaylorDiagramPlot(df, col1='obs', col2=['Model A', 'Model B'], scale=1.5)
plot.plot()

plt.title("Model Comparison Taylor Diagram")
plt.show()
