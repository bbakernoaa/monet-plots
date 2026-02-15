"""
Spatial Scatter Bias Plot (Alternative)
=======================================

**What it's for:**
This plot (intended as an alternative to Spatial Bias Scatter) visualizes model
bias at geographic locations.

**When to use:**
Use this to identify regional patterns in model errors, specifically comparing
a reference dataset (observations) with a comparison dataset (model output)
across a geographic domain.

**How to read:**
*   **Markers:** Represent data points at specific Latitude/Longitude coordinates.
*   **Color:** Represents the bias (Comparison - Reference).
*   **Interpretation:** Clusters of similar colors indicate regional systematic
    biases in the comparison dataset.

*Note: This specific example is currently a placeholder for an alternative
implementation.*
"""

import numpy as np
import pandas as pd
# from monet_plots.plots.sp_scatter_bias import SpScatterBiasPlot

# 1. Prepare sample data
np.random.seed(42)  # for reproducibility
n_points = 500

# Simulate random latitude and longitude points
lat = np.random.uniform(20, 50, n_points)
lon = np.random.uniform(-120, -70, n_points)

# Simulate reference and comparison values
reference_values = 10 + 5 * np.random.rand(n_points)
# Introduce a spatial bias: higher values in the west, lower in the east
comparison_values = reference_values + (lon / 100 + np.random.normal(0, 0.5, n_points))

df = pd.DataFrame(
    {
        "latitude": lat,
        "longitude": lon,
        "reference_value": reference_values,
        "comparison_value": comparison_values,
    }
)

# 2. Initialize and create the plot
# Example skipped: SpScatterBiasPlot module not found in this version.
