"""
Rank Histogram
==============

**What it's for:**
The Rank Histogram (also known as a Talagrand diagram) is used to evaluate the reliability
of an ensemble forecast system. It checks whether the observation is equally likely to
fall in any of the "bins" defined by the sorted ensemble members.

**When to use:**
Use this when you have an ensemble of forecasts and a corresponding observation. It is
essential for diagnosing issues with ensemble spread and bias.

**How to read:**
*   **X-axis:** The rank of the observation relative to the sorted ensemble members (0 to N).
*   **Y-axis:** Frequency or relative frequency of occurrences in each rank.
*   **Flat (Uniform):** Indicates a reliable ensemble where the observation is indistinguishable
    from the ensemble members.
*   **U-shape:** Indicates a lack of spread (the ensemble is over-confident; the observation
    frequently falls outside the ensemble range).
*   **Dome shape:** Indicates too much spread (the ensemble is under-confident; the observation
    falls in the middle more often than expected).
*   **Asymmetry (Sloping):** Indicates a systematic bias in the ensemble mean.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from monet_plots.plots.rank_histogram import RankHistogramPlot

# 1. Prepare sample data for a uniform rank distribution
np.random.seed(42)  # for reproducibility
n_ensemble_members = 10
n_forecasts = 1000

# Simulate ranks from 0 to n_ensemble_members (inclusive)
# A uniform distribution means each rank is equally likely
ranks_uniform = np.random.randint(0, n_ensemble_members + 1, n_forecasts)
df_uniform = pd.DataFrame({"rank": ranks_uniform})

# 2. Initialize and create the plot
plot = RankHistogramPlot(
    df_uniform, rank_col="rank", n_members=n_ensemble_members, figsize=(10, 6)
)
plot.plot(normalize=True)

# 3. Add titles and labels
plot.ax.set_title("Rank Histogram: Uniform Distribution (Reliable Ensemble)")
plot.ax.set_xlabel("Rank of Observation")
plot.ax.set_ylabel("Relative Frequency")

plt.tight_layout()
plt.show()
