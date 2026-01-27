"""
Rank Histogram
==============

This example demonstrates how to create a Rank Histogram.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from monet_plots.plots.rank_histogram import RankHistogramPlot

# 1. Prepare sample data for a uniform rank distribution
np.random.seed(42) # for reproducibility
n_ensemble_members = 10
n_forecasts = 1000

# Simulate ranks from 0 to n_ensemble_members (inclusive)
# A uniform distribution means each rank is equally likely
ranks_uniform = np.random.randint(0, n_ensemble_members + 1, n_forecasts)
df_uniform = pd.DataFrame({'rank': ranks_uniform})

# 2. Initialize and create the plot
plot = RankHistogramPlot(figsize=(10, 6))
plot.plot(df_uniform, rank_col='rank', n_members=n_ensemble_members, normalize=True)

# 3. Add titles and labels
plot.ax.set_title("Rank Histogram: Uniform Distribution (Reliable Ensemble)")
plot.ax.set_xlabel("Rank of Observation")
plot.ax.set_ylabel("Relative Frequency")

plt.tight_layout()
plt.show()
