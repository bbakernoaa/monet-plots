"""
Diurnal Error Plot
==================

**What it's for:**
A Diurnal Error plot visualizes how model performance (typically Mean Bias) varies
across the 24-hour daily cycle.

**When to use:**
Use this to diagnose systemic issues in the model that are linked to the time of day,
such as errors in the timing of emissions, the development of the planetary boundary
layer, or the model's response to solar radiation.

**How to read:**
*   **X-axis:** Hour of the day (0-23), often in local time or UTC.
*   **Y-axis:** Represents the error metric (e.g., Model - Observation).
*   **Interpretation:** Look for consistent peaks or troughs at specific times. For
    example, a large positive bias in the morning might indicate an issue with
    how the model handles the transition from a stable nighttime boundary layer
    to a convective daytime one.
"""

import numpy as np
import pandas as pd

from monet_plots.plots.diurnal_error import DiurnalErrorPlot

# Create dummy time series data
dates = pd.date_range("2023-01-01", periods=24 * 30, freq="h")
df = pd.DataFrame(
    {
        "time": dates,
        "obs": np.random.rand(24 * 30) * 10 + 5,
        "mod": np.random.rand(24 * 30) * 10 + 6,  # Slight bias
    }
)

# Initialize and plot
plot = DiurnalErrorPlot(df, obs_col="obs", mod_col="mod", second_dim="dayofweek")
plot.plot()
plot.save("diurnal_error_example.png")
print("Diurnal error plot saved to diurnal_error_example.png")
