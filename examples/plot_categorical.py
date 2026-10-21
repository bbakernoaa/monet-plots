"""
Categorical Plot
================

**What it's for:**
The `categorical_plot` function is a versatile tool for visualizing data that is
grouped into discrete categories (e.g., different models, sites, or seasons). It
supports various underlying plot types like bars, boxes, or violins.

**When to use:**
Use this to compare aggregate statistics across different groups. It is ideal for
showing how Mean Bias or RMSE varies between different model versions or different
geographic regions.

**How to read:**
*   **X-axis:** Represents discrete categories.
*   **Y-axis:** Represents the numerical variable being compared.
*   **Interpretation:** In a bar plot (as shown here), the height of the bar usually
    represents the mean or median of the group. Error bars (if present) show the
    variability or uncertainty within that category.
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from monet_plots.plots.categorical import categorical_plot

# 1. Prepare sample data
# Create a sample xarray DataArray with a categorical dimension
np.random.seed(42)  # for reproducibility
data = xr.DataArray(
    np.random.normal(loc=10, scale=2, size=(100, 3)),
    coords={"sample": np.arange(100), "category": ["Group A", "Group B", "Group C"]},
    dims=["sample", "category"],
    name="measurement",
)

# 2. Create a basic bar plot
# Note: categorical_plot handles conversion to dataframe internally if needed
fig, ax = categorical_plot(
    data,
    x="category",
    y="measurement",
    kind="bar",
    title="Mean Measurement per Category",
    xlabel="Category",
    ylabel="Measurement Value",
)

plt.tight_layout()
plt.show()
