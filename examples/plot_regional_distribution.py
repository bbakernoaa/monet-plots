"""
Regional Distribution Plot Example
==================================

What it's for
------------
This script demonstrates how to create a grouped boxplot comparison of regional
distributions between a model and a reference dataset, with a geospatial inset map.

When to use
-----------
Use this plot when you have data aggregated or sliced by regions (e.g., IPCC AR6
regions) and you want to compare the statistical distributions (median, quartiles)
of a variable across those regions.

How to read
-----------
Each pair of boxes represents the distribution of values within a specific region
for the model (e.g., orange) and the reference (e.g., blue). The inset map
provides geographic context.
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from monet_plots.plots import RegionalDistributionPlot

# 1. Create Mock Data
regions = ["NAU", "SAU", "AMZ", "SSA", "CAM", "WNA", "CNA", "ENA"]
n_points = 100

# Simulate Model data
da_model = xr.DataArray(
    np.random.rand(len(regions), n_points) + 0.1,
    coords={"region": regions},
    dims=("region", "point"),
    name="AOD",
)

# Simulate Reference data
da_ref = xr.DataArray(
    np.random.rand(len(regions), n_points),
    coords={"region": regions},
    dims=("region", "point"),
    name="AOD",
)

# 2. Initialize and Generate Plot
# We pass the data as a list of DataArrays and provide labels
plot = RegionalDistributionPlot(
    [da_ref, da_model],
    labels=["AERONET", "P8"],
    group_dim="region",
    var_label="AOD",
    figsize=(15, 6),
)

# Add the main boxplot
ax = plot.plot()

# 3. Add an Inset Map
# We can customize the position and add map features
plot.add_inset_map(
    inset_pos=[0.15, 0.55, 0.3, 0.35],
    coastlines=True,
    states=False,
    land=True,
    ocean=True,
)

# 4. Final adjustments and save
plt.show()
# plot.save('regional_distribution_example.png')
