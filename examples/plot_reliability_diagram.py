"""
Reliability Diagram
===================

**What it's for:**
A Reliability Diagram (or calibration curve) evaluates how well the predicted probabilities of a
probabilistic forecast system match the actual frequency of observed events.

**When to use:**
Use this diagram to assess the calibration of any probabilistic model (e.g., weather forecasts,
machine learning classifiers). It helps determine if the model is overconfident,
underconfident, or well-calibrated.

**How to read:**
*   **X-axis:** The forecast probability (often grouped into bins).
*   **Y-axis:** The observed relative frequency of the event for each probability bin.
*   **1:1 Diagonal Line:** Represents perfect reliability. If the model says there is a 70%
    chance of an event, and it occurs 70% of the time, the point will fall on this line.
*   **Points below the line:** Indicate over-forecasting or over-confidence (the predicted
    probability is higher than the actual frequency).
*   **Points above the line:** Indicate under-forecasting (the predicted probability is lower
    than the actual frequency).
*   **Histogram/Inset:** Often shows the sample frequency in each bin, providing context on
    how much data supports each point.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from monet_plots.plots.reliability_diagram import ReliabilityDiagramPlot

# 1. Prepare sample data
np.random.seed(42)  # for reproducibility
n_samples = 1000

# Simulate forecast probabilities (e.g., probability of rain)
forecast_probabilities = np.random.rand(n_samples)

# Simulate observations based on these probabilities (binary outcome)
# Introduce some unreliability: forecasts are slightly overconfident at low probs, underconfident at high probs
observations = (
    np.random.rand(n_samples) < (forecast_probabilities * 0.8 + 0.1)
).astype(int)

df = pd.DataFrame(
    {"forecast_prob": forecast_probabilities, "observed_event": observations}
)

# 2. Initialize and create the plot
plot = ReliabilityDiagramPlot(
    df,
    forecasts_col="forecast_prob",
    observations_col="observed_event",
    n_bins=10,
    figsize=(8, 8),
)
plot.plot(
    markersize=8,
    color="blue",
    label="Forecast System",
)

# 3. Add titles and labels
plot.ax.set_title("Reliability Diagram for a Probabilistic Forecast")
plot.ax.legend(loc="upper left")

plt.tight_layout()
plt.show()
