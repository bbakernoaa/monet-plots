"""
Grouped Distribution Plot Example
=================================

What it's for
------------
This script demonstrates how to create a grouped boxplot comparison of statistical
distributions across different categories or groups (e.g., seasons, regions, or
experiment sites) for multiple datasets.

When to use
-----------
Use this plot when you have data aggregated or sliced by groups and you want to
compare the statistical distributions (median, quartiles) of a variable across
those groups for multiple models or reference datasets.

How to read
-----------
Each group on the x-axis contains multiple boxes, one for each dataset (e.g., Model 1,
Model 2, and Reference). The boxes show the distribution of values within that
group.
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from monet_plots.plots import GroupedDistributionPlot

# 1. Create Mock Data for three datasets
groups = ["Group A", "Group B", "Group C", "Group D"]
n_points = 100

# Reference data
da_ref = xr.DataArray(
    np.random.rand(len(groups), n_points),
    coords={"category": groups},
    dims=("category", "point"),
    name="Observation",
)

# Model 1 data
da_m1 = xr.DataArray(
    np.random.rand(len(groups), n_points) + 0.1,
    coords={"category": groups},
    dims=("category", "point"),
    name="Model P8",
)

# Model 2 data
da_m2 = xr.DataArray(
    np.random.rand(len(groups), n_points) * 0.9,
    coords={"category": groups},
    dims=("category", "point"),
    name="Model MERRA2",
)

# 2. Initialize and Generate Plot
# We pass the data as a list of DataArrays.
# The class will automatically use the DataArray names for the legend
# and the 'category' dimension for grouping.
plot = GroupedDistributionPlot(
    [da_ref, da_m1, da_m2],
    group_dim="category",
    var_label="AOD",
    figsize=(12, 6),
)

# Add the main boxplot
ax = plot.plot()

# 3. Final adjustments and show
plt.show()
# plot.save('grouped_distribution_example.png')
