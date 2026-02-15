"""
Conditional Bias
================

**What it's for:**
A Conditional Bias plot examines how the mean bias of a model varies as a function of the
observed or modeled value itself.

**When to use:**
Use this plot to identify systematic errors that are dependent on the magnitude of the variable.
For example, it can reveal if a model consistently over-predicts low concentrations but
under-predicts high concentrations (a common issue in many environmental models).

**How to read:**
*   **X-axis:** The reference value (typically the observed value), often grouped into bins.
*   **Y-axis:** The mean bias (Model - Observation) calculated for each bin.
*   **Zero Line:** A horizontal line at zero represents no bias.
*   **Interpretation:** Points above the zero line indicate over-prediction for that range,
    while points below indicate under-prediction. The trend of the points shows how the
    model's systematic error changes across the data range.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from monet_plots.plots.conditional_bias import ConditionalBiasPlot

# 1. Prepare sample data
np.random.seed(42)  # for reproducibility
n_samples = 500

# Simulate observations
observations = np.random.normal(loc=10, scale=3, size=n_samples)

# Simulate forecasts with a bias that depends on the observation value:
# - For lower observations, forecast is slightly higher (positive bias)
# - For higher observations, forecast is slightly lower (negative bias)
# - Add some random noise
forecasts = (
    observations
    + (0.5 - 0.1 * observations)
    + np.random.normal(loc=0, scale=0.5, size=n_samples)
)

# Ensure data is in a DataFrame
df = pd.DataFrame({"observations": observations, "forecasts": forecasts})

# 2. Initialize and create the plot
plot = ConditionalBiasPlot(
    df, obs_col="observations", fcst_col="forecasts", n_bins=15, figsize=(10, 6)
)
plot.plot()

# 3. Add titles and labels
plot.ax.set_title("Conditional Bias Plot (Forecast vs. Observation)")
plot.ax.set_xlabel("Observed Value")
plot.ax.set_ylabel("Mean Bias (Forecast - Observation)")

plt.tight_layout()
plt.show()
