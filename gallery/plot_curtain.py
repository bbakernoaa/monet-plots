"""
Curtain Plot
============

**What it's for:**
A Curtain plot (or time-height cross-section) visualizes how a vertical profile changes over
time at a fixed location, or along a moving path (like a flight track).

**When to use:**
Use this to monitor the evolution of the planetary boundary layer, track the arrival and
dispersion of a smoke plume, or analyze the vertical structure of a weather system as it
passes over a station.

**How to read:**
*   **X-axis:** Typically represents Time (or distance along a track).
*   **Y-axis:** Represents Altitude or Pressure.
*   **Color:** Represents the magnitude of the variable being plotted (e.g., PM2.5
    concentration, potential temperature).
*   **Interpretation:** Look for temporal trends at specific heights or the movement of
    features (like an inversion layer) upward or downward over time.
"""

import numpy as np
import pandas as pd
import xarray as xr

from monet_plots.plots.curtain import CurtainPlot

# Create dummy 2D data
times = pd.date_range("2023-01-01", periods=24, freq="h")
levels = np.linspace(1000, 100, 10)
data = np.exp(-((np.arange(24) - 12) ** 2) / 20)[:, np.newaxis] * np.exp(
    -((np.arange(10) - 5) ** 2) / 10
)
da = xr.DataArray(
    data.T, coords=[levels, times], dims=["level", "time"], name="concentration"
)

# Initialize and plot
plot = CurtainPlot(da)
plot.plot(kind="contourf", cmap="viridis")
plot.save("curtain_example.png")
print("Curtain plot saved to curtain_example.png")
