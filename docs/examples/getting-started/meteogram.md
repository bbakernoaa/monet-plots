# Meteogram Plots

Meteograms are specialized time series plots that display the evolution of multiple meteorological (or other environmental) variables over time, typically stacked vertically. They are invaluable for quickly assessing atmospheric conditions, identifying trends, and understanding the interplay between different parameters at a specific location.

## Prerequisites

-   Basic Python knowledge
-   Understanding of numpy and pandas
-   MONET Plots installed (`pip install monet_plots`)
-   Matplotlib installed (`pip install matplotlib`)

## Plotting Workflow

To create a meteogram with `monet_plots`:

1.  **Prepare Data**: Your data should be in a `pandas.DataFrame` with a `DatetimeIndex` (or a column that can be set as the index) and columns representing the variables you want to plot.
2.  **Initialize `Meteogram`**: Create an instance of the `Meteogram` class, passing your DataFrame and a list of the variable names to be plotted.
3.  **Call `plot` method**: Generate the stacked time series plots.
4.  **Customize (Optional)**: While `Meteogram` handles much of the layout, you can still add overall titles or adjust individual subplot properties if needed.
5.  **Display/Save**: Show the plot using `plt.show()` or save it to a file.

## Example 1: Basic Meteogram Plot

Let's create a basic meteogram using synthetic time series data for temperature, humidity, and pressure.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from monet_plots.plots.meteogram import Meteogram

# 1. Prepare sample data
np.random.seed(42) # for reproducibility
dates = pd.date_range('2023-01-01 00:00', periods=24, freq='h')

# Simulate temperature with a diurnal cycle
temperature = 20 + 5 * np.sin(np.linspace(0, 2 * np.pi, 24)) + np.random.normal(0, 0.5, 24)
# Simulate humidity
humidity = 70 - 10 * np.cos(np.linspace(0, 2 * np.pi, 24)) + np.random.normal(0, 1, 24)
# Simulate pressure
pressure = 1012 + 3 * np.sin(np.linspace(0, 2 * np.pi, 24) + np.pi/4) + np.random.normal(0, 0.2, 24)

df = pd.DataFrame({
    'Temperature': temperature,
    'Humidity': humidity,
    'Pressure': pressure
}, index=dates)

# 2. Initialize and create the plot
plot = Meteogram(df=df, variables=['Temperature', 'Humidity', 'Pressure'], figsize=(12, 9))
plot.plot(linewidth=1.5, marker='o', markersize=3) # Plot with lines and markers

# Add an overall title to the figure
plot.fig.suptitle("Synthetic Meteogram for a 24-hour Period", fontsize=16)

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
plt.show()
```

### Expected Output

A figure will be displayed containing three vertically stacked subplots. Each subplot will show the time series of one variable ('Temperature', 'Humidity', 'Pressure') over the 24-hour period. The x-axis (time) will be shared across all subplots, and each subplot will have its own y-axis labeled with the variable name. The plots will show lines with markers, illustrating the diurnal variations of each simulated variable.

### Key Concepts

-   **`Meteogram`**: The class used to generate stacked time series plots for multiple variables.
-   **`df`**: The input `pandas.DataFrame` must have a `DatetimeIndex` for the time axis.
-   **`variables`**: A list of strings corresponding to the column names in the DataFrame that you wish to plot. Each variable will get its own subplot.
-   **Stacked Subplots**: The design of the meteogram automatically arranges each variable's time series in a separate subplot, sharing the x-axis for easy comparison of temporal alignment.
-   **`plt.tight_layout()`**: Important for adjusting subplot parameters for a tight layout, especially with titles and labels.
