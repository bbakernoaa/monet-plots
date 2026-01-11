"""
Rank Histogram
==============

This example demonstrates the Rank Histogram (also known as a Talagrand
diagram), which is used to assess the spread of an ensemble forecast.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from monet_plots.plots.rank_histogram import RankHistogram

# 1. Create synthetic ensemble forecast data and corresponding observations
np.random.seed(42)
n_samples = 1000
n_members = 11  # Number of ensemble members

# Generate ensemble forecasts
ensemble_forecasts = np.random.randn(n_samples, n_members) * 5 + 280

# Generate corresponding "truth" observations
# A well-calibrated ensemble should have the truth fall within its spread
observations = np.random.randn(n_samples) * 4.5 + 280.5

# Combine into a DataFrame
df = pd.DataFrame(ensemble_forecasts, columns=[f"member_{i}" for i in range(n_members)])
df['observation'] = observations

# 2. Create and display the plot
fig, ax = plt.subplots(figsize=(8, 6))
plot = RankHistogram(ax=ax, data=df)
plot.plot(
    obs_col="observation",
    ensemble_cols=[f"member_{i}" for i in range(n_members)],
    title="Rank Histogram of Ensemble Forecast",
)
plt.show()
