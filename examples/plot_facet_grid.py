"""
Facet Grid Plot
===============

**What it's for:**
A Facet Grid allows you to create a matrix of subplots (facets) based on categorical
variables in your dataset. It is a powerful way to visualize the same relationship
across different subsets of data.

**When to use:**
Use this to explore how a relationship (e.g., between model and observation) changes
across different sites, months, variables, or experiment groups. It enables
"small multiple" visualizations that are much easier to compare than separate plots.

**How to read:**
*   **Columns/Rows:** Represent different levels of a categorical variable (e.g.,
    each column is a different city).
*   **Subplots:** Each facet contains a plot of the same type (e.g., a scatter plot).
*   **Interpretation:** Compare the trends, slopes, or distributions across the
    facets to identify inconsistencies or regional/temporal differences.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from monet_plots.plots.facet_grid import FacetGridPlot

# 1. Prepare sample data
np.random.seed(42)  # for reproducibility
n_samples_per_category = 100
categories = ["Group A", "Group B", "Group C"]

data_list = []
for i, cat in enumerate(categories):
    x = np.random.normal(loc=i * 5, scale=2, size=n_samples_per_category)
    y = 0.5 * x + np.random.normal(loc=0, scale=1, size=n_samples_per_category) + i * 3
    temp_df = pd.DataFrame({"x_data": x, "y_data": y, "category": cat})
    data_list.append(temp_df)

df = pd.concat(data_list).reset_index(drop=True)

# 2. Initialize FacetGridPlot
# We'll create a column for each 'category'
fg_plot = FacetGridPlot(data=df, col="category", height=4, aspect=1.2)

# 3. Map a scatter plot to each facet
fg_plot.grid.map(plt.scatter, "x_data", "y_data", alpha=0.7, s=50, edgecolor="w")

# 4. Set titles and labels
fg_plot.set_titles(col_template="Category: {col_name}")
fg_plot.grid.set_xlabels("X-axis Data")
fg_plot.grid.set_ylabels("Y-axis Data")
fg_plot.fig.suptitle(
    "Scatter Plot across Categories", y=1.02
)  # y adjusts title position

plt.tight_layout()
plt.show()
