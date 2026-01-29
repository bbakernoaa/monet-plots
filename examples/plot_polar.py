"""
Bivariate Polar Plot
====================

**What it's for:**
A Bivariate Polar plot (sometimes called a polar plot or bivariate plot) visualizes
how a variable (e.g., pollutant concentration) varies as a function of wind speed
and wind direction.

**When to use:**
Use this to identify the potential source directions and distances (using wind speed
as a proxy for distance/source type) of observed concentrations at a monitoring
site. High concentrations at low wind speeds often suggest local sources, while
high concentrations at high wind speeds suggest long-range transport.

**How to read:**
*   **Angle:** Represents the Wind Direction (0-360 degrees).
*   **Radial Distance:** Represents the Wind Speed.
*   **Color:** Represents the magnitude of the variable (e.g., concentration).
*   **Interpretation:** Bright clusters indicate specific wind conditions (direction
    and speed) associated with high values of the variable.
"""

import pandas as pd
import numpy as np
from monet_plots.plots.polar import BivariatePolarPlot

# Create dummy wind dependent data
n = 1000
ws = np.random.gamma(2, 2, n)
wd = np.random.uniform(0, 360, n)
# Concentration higher when wind is from the East (90 deg) and low speed
conc = 10 * np.exp(-((wd - 90) ** 2) / 1000) * np.exp(-ws / 5) + np.random.rand(n)

df = pd.DataFrame({"ws": ws, "wd": wd, "conc": conc})

# Initialize and plot
plot = BivariatePolarPlot(df, ws_col="ws", wd_col="wd", val_col="conc")
plot.plot(n_bins_ws=15, n_bins_wd=36)
plot.save("polar_example.png")
print("Bivariate polar plot saved to polar_example.png")
