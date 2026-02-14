"""
Meteogram
=========

**What it's for:**
A Meteogram is a time series plot that displays multiple meteorological variables (e.g.,
temperature, pressure, humidity, wind) for a single geographical location.

**When to use:**
Use this to visualize the evolution of local weather conditions over a specific period.
It is commonly used for weather forecasting, climate monitoring, and analyzing site-specific
observational data.

**How to read:**
*   **X-axis:** Represents time (usually UTC or local time).
*   **Y-axes:** Each variable is typically plotted on its own axis or in a stacked sub-plot.
*   **Interpretation:** Look for correlations between variables (e.g., a drop in pressure
    followed by a change in wind direction and temperature, indicating a frontal passage).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from monet_plots.plots.meteogram import Meteogram

# 1. Prepare sample data
np.random.seed(42)  # for reproducibility
dates = pd.date_range("2023-01-01 00:00", periods=24, freq="h")

# Simulate temperature with a diurnal cycle
temperature = (
    20 + 5 * np.sin(np.linspace(0, 2 * np.pi, 24)) + np.random.normal(0, 0.5, 24)
)
# Simulate humidity
humidity = 70 - 10 * np.cos(np.linspace(0, 2 * np.pi, 24)) + np.random.normal(0, 1, 24)
# Simulate pressure
pressure = (
    1012
    + 3 * np.sin(np.linspace(0, 2 * np.pi, 24) + np.pi / 4)
    + np.random.normal(0, 0.2, 24)
)

df = pd.DataFrame(
    {"Temperature": temperature, "Humidity": humidity, "Pressure": pressure},
    index=dates,
)

# 2. Initialize and create the plot
plot = Meteogram(
    df=df, variables=["Temperature", "Humidity", "Pressure"], figsize=(12, 9)
)
plot.plot(linewidth=1.5, marker="o", markersize=3)  # Plot with lines and markers

# Add an overall title to the figure
plot.fig.suptitle("Synthetic Meteogram for a 24-hour Period", fontsize=16)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to prevent title overlap
plt.show()
