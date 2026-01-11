"""
Facet Grid Plot
===============

This example demonstrates the FacetGrid plot, which allows for creating a grid
of subplots based on different categories or values in the dataset.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from monet_plots.plots.facet_grid import FacetGridPlot

# 1. Create a synthetic dataset with multiple categories
np.random.seed(42)
n_samples = 500
df = pd.DataFrame({
    'x': np.random.randn(n_samples),
    'y': np.random.randn(n_samples) + 2,
    'category': np.random.choice(['A', 'B', 'C'], n_samples),
    'season': np.random.choice(['Winter', 'Summer'], n_samples)
})

# 2. Create and display the plot
# We will create a grid of scatter plots, with each column representing a
# 'season' and each row representing a 'category'.
plot = FacetGridPlot(data=df, row='category', col='season')
plot.map_dataframe(sns.scatterplot, 'x', 'y')
plot.grid.add_legend()
plt.suptitle('Facet Grid of Scatter Plots by Season and Category', y=1.02)
plt.show()
