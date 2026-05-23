"""
Time Series Statistics Plot
===========================

**What it's for:**
The `TimeSeriesStatsPlot` class visualizes time series data with additional
statistical context, such as means, medians, or shaded uncertainty regions.

**When to use:**
Use this when you want to show the overall trend of a dataset along with its
variability, for example, plotting the mean of several model runs with a
shaded area representing the spread.

**How to read:**
*   **Main Line:** Typically represents the mean or median value.
*   **Shaded Region:** Represents a measure of spread (e.g., standard deviation,
    min/max, or interquartile range).
*   **X-axis:** Time.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from monet_plots.plots.timeseries import TimeSeriesStatsPlot

# 1. Create dummy time series data for multiple "members"
dates = pd.date_range("2023-01-01", periods=24, freq="h")
n_members = 10
data = np.zeros((len(dates), n_members))

# Base signal
base = 10 + 5 * np.sin(np.linspace(0, 2 * np.pi, 24))

for i in range(n_members):
    data[:, i] = base + np.random.normal(0, 2, 24)

df = pd.DataFrame(data, index=dates)
df.columns = [f"member_{i}" for i in range(n_members)]
df["time"] = dates

# 2. Initialize and plot
# TimeSeriesStatsPlot requires col1 (obs) and col2 (model)
plot = TimeSeriesStatsPlot(
    df=df,
    col1="member_0",
    col2=[f"member_{i}" for i in range(1, n_members)],
    figsize=(12, 6),
)

# In this example, we manually plot the mean and spread using standard matplotlib on the provided axis
# Note: TimeSeriesStatsPlot is often used as a base for more complex statistical time series
mean_vals = df.iloc[:, :n_members].mean(axis=1)
std_vals = df.iloc[:, :n_members].std(axis=1)

plot.ax.plot(df["time"], mean_vals, label="Ensemble Mean", color="blue", linewidth=2)
plot.ax.fill_between(
    df["time"],
    mean_vals - std_vals,
    mean_vals + std_vals,
    alpha=0.3,
    color="blue",
    label="1 Std Dev",
)

# 3. Add titles and labels
plot.ax.set_title("Time Series with Ensemble Spread")
plot.ax.set_ylabel("Value")
plot.ax.legend()

plt.show()
