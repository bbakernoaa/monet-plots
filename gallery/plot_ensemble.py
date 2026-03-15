"""
Ensemble Spread-Skill Plot
==========================

**What it's for:**
The Spread-Skill plot evaluates the reliability of an ensemble forecast system by
comparing the ensemble spread (the standard deviation among ensemble members) to
the skill (the actual error, usually RMSE, of the ensemble mean).

**When to use:**
Use this to determine if your ensemble is well-calibrated. A perfectly calibrated
ensemble should have a spread that matches the expected error of the ensemble mean.

**How to read:**
*   **X-axis:** Ensemble Spread.
*   **Y-axis:** Ensemble Skill (RMSE of the mean).
*   **1:1 Line:** Represents a perfectly calibrated ensemble.
*   **Points above the line:** Indicate the ensemble is "under-dispersive" (too
    little spread; the model is overconfident).
*   **Points below the line:** Indicate the ensemble is "over-dispersive" (too
    much spread; the model is under-confident).
"""

import matplotlib.pyplot as plt
import numpy as np

from monet_plots.plots.ensemble import SpreadSkillPlot

# 1. Prepare sample data
np.random.seed(42)  # for reproducibility

# Simulate ensemble spread (standard deviation of ensemble members)
# Values typically range from 0 upwards
spread_data = np.random.uniform(0.5, 3.0, 50)

# Simulate ensemble skill (RMSE of ensemble mean)
# For a reliable ensemble, skill should be close to spread.
# We'll add some noise around the spread to simulate real-world data.
skill_data = spread_data + np.random.normal(0, 0.5, 50)
# Ensure skill is non-negative
skill_data[skill_data < 0] = 0.1

# 2. Initialize and create the plot
plot = SpreadSkillPlot(spread=spread_data, skill=skill_data, figsize=(8, 8))
plot.plot(color="blue", alpha=0.7, s=50, label="Ensemble Data")

# The plot method automatically adds labels, title, and a 1:1 line.
# You can further customize if needed.
plot.ax.legend()  # Display the legend for the scatter points

plt.tight_layout()
plt.show()
