"""
Rank Histogram
==============

This example demonstrates the Rank Histogram (also known as a Talagrand
diagram), which is used to assess the spread of an ensemble forecast.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from monet_plots.plots.rank_histogram import RankHistogramPlot

# 1. Create synthetic ensemble forecast data and corresponding observations
np.random.seed(42)
n_samples = 1000
n_members = 11  # Number of ensemble members

# Generate ensemble forecasts
ensemble_forecasts = np.random.randn(n_samples, n_members) * 5 + 280

# Generate corresponding "truth" observations
observations = np.random.randn(n_samples) * 4.5 + 280.5


# 2. Calculate the rank of the observation within the ensemble
all_data = np.hstack([ensemble_forecasts, observations[:, np.newaxis]])
ranks = np.apply_along_axis(lambda x: np.searchsorted(np.sort(x), x[-1]), 1, all_data)
rank_df = pd.DataFrame({"rank": ranks})


# 3. Create and display the plot
fig, ax = plt.subplots(figsize=(8, 6))
plot = RankHistogramPlot(ax=ax, fig=fig)
plot.plot(data=rank_df, rank_col="rank", n_members=n_members)
ax.set_title("Rank Histogram of Ensemble Forecast")
plt.show()
