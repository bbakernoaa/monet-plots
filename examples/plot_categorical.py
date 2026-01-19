"""
Categorical
===========

This example demonstrates how to create a Categorical.
"""

import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from monet_plots.plots.categorical import categorical_plot

# 1. Prepare sample data
# Create a sample xarray DataArray with a categorical dimension
np.random.seed(42) # for reproducibility
data = xr.DataArray(
    np.random.normal(loc=10, scale=2, size=(100, 3)),
    coords={
        'sample': np.arange(100),
        'category': ['Group A', 'Group B', 'Group C']
    },
    dims=['sample', 'category'],
    name='measurement'
)

# 2. Create a basic bar plot
# Note: categorical_plot handles conversion to dataframe internally if needed
fig, ax = categorical_plot(
    data,
    x='category',
    y='measurement',
    kind='bar',
    title='Mean Measurement per Category',
    xlabel='Category',
    ylabel='Measurement Value'
)

plt.tight_layout()
plt.show()
