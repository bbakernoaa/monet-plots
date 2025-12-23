# Ensemble Spread-Skill Plots

Ensemble spread-skill plots are a crucial tool for evaluating the reliability of ensemble forecasts. They compare the ensemble spread (a measure of forecast uncertainty, typically the standard deviation of ensemble members) against the ensemble skill (a measure of forecast error, typically the root mean squared error of the ensemble mean). For a perfectly reliable ensemble, the spread should be equal to the skill, meaning the forecast uncertainty accurately reflects the forecast error.

## Prerequisites

-   Basic Python knowledge
-   Understanding of numpy
-   MONET Plots installed (`pip install monet_plots`)
-   Matplotlib installed (`pip install matplotlib`)

## Plotting Workflow

The workflow for creating a spread-skill plot involves:

1.  **Prepare Data**: Generate or load your ensemble spread and skill data, typically as 1D arrays.
2.  **Initialize `SpreadSkillPlot`**: Create an instance of the `SpreadSkillPlot` class, passing the spread and skill data.
3.  **Call `plot` method**: Generate the scatter plot.
4.  **Customize (Optional)**: Add titles, labels, and other visual enhancements (though `SpreadSkillPlot` provides defaults).
5.  **Display/Save**: Show the plot using `plt.show()` or save it to a file.

## Example 1: Basic Spread-Skill Plot

Let's create a basic spread-skill plot using synthetic data to demonstrate its usage.

```python
import numpy as np
import matplotlib.pyplot as plt
from monet_plots.plots.ensemble import SpreadSkillPlot

# 1. Prepare sample data
np.random.seed(42) # for reproducibility

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
plot.plot(color='blue', alpha=0.7, s=50, label='Ensemble Data')

# The plot method automatically adds labels, title, and a 1:1 line.
# You can further customize if needed.
plot.ax.legend() # Display the legend for the scatter points

plt.tight_layout()
plt.show()
```

### Expected Output

A scatter plot will be displayed with "Ensemble Spread (Standard Deviation)" on the x-axis and "Ensemble Error (RMSE)" on the y-axis. Each point represents a pair of spread and skill values. A dashed black line, representing the 1:1 relationship, will also be present. Points falling close to this line indicate good ensemble reliability. In this synthetic example, points should generally cluster around the 1:1 line with some scatter.

### Key Concepts

-   **`SpreadSkillPlot`**: The class used to generate spread-skill plots.
-   **`spread`**: An array-like object containing the ensemble spread values.
-   **`skill`**: An array-like object containing the ensemble skill (error) values.
-   **1:1 Line**: A diagonal line where `spread = skill`. Points on this line indicate a perfectly reliable ensemble.
-   **Ensemble Reliability**: The closer the data points are to the 1:1 line, the more reliable the ensemble forecast is considered to be, as its stated uncertainty (spread) matches its actual error (skill).
