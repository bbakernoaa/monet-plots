"""
Time Series Plot
================

**What it's for:**
A Time Series plot visualizes how one or more variables change over a continuous
temporal interval.

**When to use:**
Use this for monitoring data, model output at a specific location, or area-averaged
values over time. It is the primary tool for identifying trends, diurnal cycles,
seasonal patterns, and episodic events.

**How to read:**
*   **X-axis:** Represents Time (UTC or local).
*   **Y-axis:** Represents the value of the variable.
*   **Interpretation:** Look for temporal trends, variability, and the timing of
    maximum/minimum values. Multiple lines can be used to compare models against
    observations or different model scenarios.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from monet_plots.plots.timeseries import TimeSeriesPlot

dates = pd.date_range("2023-01-01", periods=100, freq="D")
values = np.cumsum(np.random.normal(0, 1, 100)) + 50
df = pd.DataFrame({"time": dates, "values": values})
plot = TimeSeriesPlot(df=df, figsize=(12, 6))
plot.plot(x="time", y="values", title="Daily Time Series", ylabel="Temperature (°C)")
plt.show()
