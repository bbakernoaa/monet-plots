"""
Reliability Diagram
===================

This example demonstrates how to create a Reliability Diagram.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from monet_plots.plots.reliability_diagram import ReliabilityDiagramPlot

# 1. Prepare sample data
np.random.seed(42) # for reproducibility
n_samples = 1000

# Simulate forecast probabilities (e.g., probability of rain)
forecast_probabilities = np.random.rand(n_samples)

# Simulate observations based on these probabilities (binary outcome)
# Introduce some unreliability: forecasts are slightly overconfident at low probs, underconfident at high probs
observations = (np.random.rand(n_samples) < (forecast_probabilities * 0.8 + 0.1)).astype(int)

df = pd.DataFrame({
    'forecast_prob': forecast_probabilities,
    'observed_event': observations
})

# 2. Initialize and create the plot
plot = ReliabilityDiagramPlot(figsize=(8, 8))
plot.plot(
    df,
    forecasts_col='forecast_prob',
    observations_col='observed_event',
    n_bins=10,
    markersize=8,
    color='blue',
    label='Forecast System'
)

# 3. Add titles and labels
plot.ax.set_title("Reliability Diagram for a Probabilistic Forecast")
plot.ax.legend(loc='upper left')

plt.tight_layout()
plt.show()
