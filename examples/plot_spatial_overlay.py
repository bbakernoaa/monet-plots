"""
Spatial Overlay Plot
====================

**What it's for:**
The `SpatialOverlayPlot` class allows you to overlay point-based observations
(like monitoring sites) on top of a 2D gridded field (like model output).

**When to use:**
Use this to visually compare model fields with observations in a single map.
A shared colorscale is typically used to facilitate direct comparison.

**How to read:**
*   **Contoured/Shaded Field:** Represents the continuous model data.
*   **Scatter Points:** Represent the discrete observational measurements.
*   **Comparison:** If the point color blends with the background, the model
    and observations are in close agreement.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from monet_plots.plots.spatial_overlay import SpatialOverlayPlot

# 1. Create dummy gridded model data
lat = np.linspace(25, 50, 50)
lon = np.linspace(-125, -70, 60)
model_data = 20 + 30 * (
    np.sin(lat[:, np.newaxis] / 10.0) * np.cos(lon[np.newaxis, :] / 10.0) + 1
)
model_da = xr.DataArray(
    model_data,
    coords={"lat": lat, "lon": lon},
    dims=["lat", "lon"],
    name="model_ozone",
)

# 2. Create dummy point observations
n_obs = 30
obs_df = pd.DataFrame(
    {
        "latitude": np.random.uniform(30, 45, n_obs),
        "longitude": np.random.uniform(-120, -75, n_obs),
        "obs_value": np.random.uniform(20, 80, n_obs),
    }
)

# 3. Initialize and plot
# SpatialOverlayPlot overlays the points on the gridded data
plot = SpatialOverlayPlot(
    model_da,
    obs_df,
    obs_col="obs_value",
    figsize=(12, 8),
    states=True,
    coastlines=True,
    extent=[-126, -69, 24, 51],
)
plot.plot(cmap="viridis", vmin=20, vmax=80, s=100, edgecolor="white", linewidth=1.5)

# 4. Add title
plot.ax.set_title("Model Ozone Field Overlayed with Station Observations")
plt.show()
