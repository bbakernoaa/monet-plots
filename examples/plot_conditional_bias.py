"""
Conditional Bias
================

This example demonstrates how to create a Conditional Bias.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from monet_plots.plots.conditional_bias import ConditionalBiasPlot

# 1. Prepare sample data
np.random.seed(42) # for reproducibility
n_samples = 500

# Simulate observations
observations = np.random.normal(loc=10, scale=3, size=n_samples)

# Simulate forecasts with a bias that depends on the observation value:
# - For lower observations, forecast is slightly higher (positive bias)
# - For higher observations, forecast is slightly lower (negative bias)
# - Add some random noise
forecasts = observations + (0.5 - 0.1 * observations) + np.random.normal(loc=0, scale=0.5, size=n_samples)

# Ensure data is in a DataFrame
df = pd.DataFrame({'observations': observations, 'forecasts': forecasts})

# 2. Initialize and create the plot
plot = ConditionalBiasPlot(figsize=(10, 6))
plot.plot(df, obs_col='observations', fcst_col='forecasts', n_bins=15)

# 3. Add titles and labels
plot.ax.set_title("Conditional Bias Plot (Forecast vs. Observation)")
plot.ax.set_xlabel("Observed Value")
plot.ax.set_ylabel("Mean Bias (Forecast - Observation)")

plt.tight_layout()
plt.show()
