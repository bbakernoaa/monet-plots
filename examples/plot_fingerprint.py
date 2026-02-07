"""
Fingerprint Plot
================

**What it's for:**
A Fingerprint plot is a 2D heatmap used to identify temporal patterns and cycles in a
time series, such as diurnal (daily) and seasonal variations.

**When to use:**
Use this to analyze long-term monitoring data (e.g., air quality, temperature, energy usage).
It is excellent for revealing when high-concentration events typically occur—for instance,
during morning rush hour or on specific days of the year.

**How to read:**
*   **X-axis:** Typically represents the Hour of the Day (0-23).
*   **Y-axis:** Typically represents the Day of the Year, Month, or Date.
*   **Color:** Represents the magnitude of the variable being analyzed.
*   **Interpretation:** Look for vertical bands (indicating consistent diurnal patterns
    across the year) or horizontal bands (indicating seasonal patterns). Bright spots
    highlight specific times and days with unusually high or low values.
"""

import pandas as pd
import numpy as np
from monet_plots.plots.fingerprint import FingerprintPlot

# Create dummy data showing a diurnal/seasonal pattern
dates = pd.date_range("2023-01-01", periods=24 * 365, freq="h")
hours = dates.hour
doy = dates.dayofyear
val = (
    np.sin(2 * np.pi * hours / 24)
    + np.sin(2 * np.pi * doy / 365)
    + np.random.randn(24 * 365) * 0.1
)

df = pd.DataFrame({"time": dates, "concentration": val})

# Initialize and plot
plot = FingerprintPlot(df, val_col="concentration", x_scale="hour", y_scale="dayofyear")
plot.plot(cmap="magma")
plot.save("fingerprint_example.png")
print("Fingerprint plot saved to fingerprint_example.png")
