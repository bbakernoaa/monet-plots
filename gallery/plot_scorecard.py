"""
Scorecard
=========

**What it's for:**
The Scorecard is a tabular visualization that provides a high-level overview of model
performance across multiple variables, metrics, lead times, or locations. It uses color-coding
to highlight areas of strength and weakness.

**When to use:**
Use this when you have a large number of performance statistics to present simultaneously.
It is particularly effective for comparing a new model version against a baseline, or
for identifying which meteorological variables or forecast horizons are most problematic.

**How to read:**
*   **Rows/Columns:** Represent different dimensions of the evaluation (e.g., Variable vs.
    Forecast Lead Time).
*   **Cell Color:** Indicates the performance level. Typically, a diverging colormap is
    used where green represents improvement (or good performance) and red represents
    degradation (or poor performance) relative to a benchmark.
*   **Interpretation:** Look for patterns in the colors. For example, a whole row of red
    might indicate a systemic issue with a specific variable, while a column of red might
    indicate a drop in performance at a specific lead time.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from monet_plots.plots.scorecard import ScorecardPlot
from monet_plots.style import set_style

# 1. Prepare sample data in long format
np.random.seed(42)  # for reproducibility

variables = ["Temperature", "Humidity", "Wind Speed", "Pressure"]
lead_times = ["+06h", "+12h", "+18h", "+24h", "+36h"]

data_list = []
for var in variables:
    for lt in lead_times:
        # Simulate a metric value (e.g., RMSE difference from baseline)
        # Values around 0 mean similar performance, positive means worse, negative means better
        metric_value = np.random.normal(loc=0, scale=0.5)
        if var == "Temperature" and lt == "+06h":
            metric_value = -1.2  # Example of good performance
        elif var == "Pressure" and lt == "+36h":
            metric_value = 1.5  # Example of poor performance

        data_list.append(
            {"Variable": var, "Lead Time": lt, "Metric Value": metric_value}
        )

df = pd.DataFrame(data_list)

# 2. Initialize and create the plot
plot = ScorecardPlot(figsize=(10, 7))
plot.plot(
    df,
    x_col="Lead Time",
    y_col="Variable",
    val_col="Metric Value",
    cmap="RdYlGn_r",  # Red-Yellow-Green colormap, reversed so green is good (negative values)
    center=0,  # Center the colormap at 0
    linewidths=0.5,  # Add lines between cells
    linecolor="black",
)

# 3. Add titles and labels (plot.plot sets default title and labels)
plot.ax.set_title("Model Performance Scorecard (RMSE Difference)", fontsize=14)
plot.ax.set_xlabel("Forecast Lead Time")
plot.ax.set_ylabel("Meteorological Variable")

plt.tight_layout()

# ---------------------------------------------------------
# 4. WeatherMesh-style Scorecard
# ---------------------------------------------------------

# Prepare data for WeatherMesh-style scorecard (e.g. comparing model vs observation across cities)
cities = [
    "Atlanta, GA",
    "Boston, MA",
    "Chicago, IL",
    "Houston, TX",
    "Miami, FL",
    "New York, NY",
]
data_wm = []
for city in cities:
    for lt in [1, 3, 5, 7, 10, 14]:
        mod = np.random.uniform(1, 10)
        obs = np.random.uniform(1, 10)
        diff = mod - obs
        data_wm.append(
            {"City": city, "Lead Time": lt, "Model": mod, "Obs": obs, "Diff": diff}
        )

df_wm = pd.DataFrame(data_wm)

# Apply weathermesh style
set_style("weathermesh")

# Initialize and create the WeatherMesh-style plot
plot_wm = ScorecardPlot()
plot_wm.plot(
    df_wm,
    x_col="Lead Time",
    y_col="City",
    val_col="Diff",
    annot_cols=["Model", "Obs"],
    cbar_labels=("Model Better", "Baseline Better"),
    key_text="Model RMSE | Baseline RMSE",
    title="Forecast Performance in Global Cities",
    cmap="RdBu_r",
    center=0,
)

# Move Lead Time label to bottom center
plot_wm.ax.set_xlabel("Forecast Lead Time (Days)")

plt.show()
