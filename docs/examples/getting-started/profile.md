# Profile Plots

Profile plots are used to visualize the variation of one or more variables along a single dimension, often representing vertical profiles in atmospheric or oceanic sciences, or cross-sections in other fields. MONET Plots' `ProfilePlot` class supports both 1D line plots and 2D contour plots for visualizing such data.

## Prerequisites

-   Basic Python knowledge
-   Understanding of numpy
-   MONET Plots installed (`pip install monet_plots`)
-   Matplotlib installed (`pip install matplotlib`)

## Plotting Workflow

To create a profile plot with `monet_plots`:

1.  **Prepare Data**: Organize your data into `numpy` arrays for the x-axis, y-axis, and optionally a z-axis for contour plots.
2.  **Initialize `ProfilePlot`**: Create an instance of the `ProfilePlot` class, passing the `x` and `y` arrays, and optionally the `z` array.
3.  **Call `plot` method**: Generate the profile plot.
4.  **Customize (Optional)**: Add titles, labels, or other visual enhancements.
5.  **Display/Save**: Show the plot using `plt.show()` or save it to a file.

## Example 1: Basic 1D Profile Plot

Let's create a simple 1D profile plot showing how temperature changes with altitude.

```python
import numpy as np
import matplotlib.pyplot as plt
from monet_plots.plots.profile import ProfilePlot

# 1. Prepare sample 1D data
altitude = np.linspace(0, 10000, 100) # meters
temperature = 20 - 0.0065 * altitude + 5 * np.sin(altitude / 1000) # degrees Celsius

# 2. Initialize and create the plot
plot = ProfilePlot(x=temperature, y=altitude, figsize=(7, 9))
plot.plot(color='red', linewidth=2, label='Temperature Profile')

# 3. Add titles and labels
plot.ax.set_title("Atmospheric Temperature Profile")
plot.ax.set_xlabel("Temperature (°C)")
plot.ax.set_ylabel("Altitude (m)")
plot.ax.legend()
plot.ax.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
```

### Expected Output

A 1D line plot will be displayed with "Temperature (°C)" on the x-axis and "Altitude (m)" on the y-axis. A red line will show the simulated temperature variation with increasing altitude, including some sinusoidal fluctuations. A legend will indicate "Temperature Profile".

### Key Concepts

-   **`ProfilePlot`**: The class used for creating profile visualizations.
-   **`x` and `y`**: `numpy` arrays representing the data for the horizontal and vertical axes, respectively.
-   **1D Line Plot**: When only `x` and `y` are provided, `ProfilePlot` defaults to a standard line plot.

## Example 2: 2D Contour Profile Plot

When you have data that varies across two dimensions (e.g., a cross-section of a field), you can use `ProfilePlot` to create a contour plot. Here, `x` and `y` define the grid, and `z` provides the values for the contours.

```python
import numpy as np
import matplotlib.pyplot as plt
from monet_plots.plots.profile import ProfilePlot

# 1. Prepare sample 2D data for a cross-section
np.random.seed(42) # for reproducibility

# Define horizontal and vertical coordinates
horizontal_distance = np.linspace(0, 100, 50) # km
vertical_height = np.linspace(0, 5, 30) # km

# Create a 2D grid
X, Y = np.meshgrid(horizontal_distance, vertical_height)

# Simulate a 2D field (e.g., pollutant concentration)
# Concentration varies with distance and height, with a plume-like structure
Z = (np.exp(-((X - 50)**2 / (2 * 15**2) + (Y - 2)**2 / (2 * 1**2))) * 100
     + np.random.normal(0, 5, X.shape))
Z[Z < 0] = 0 # Ensure non-negative concentrations

# 2. Initialize and create the contour plot
plot = ProfilePlot(x=X, y=Y, z=Z, figsize=(10, 7))
plot.plot(levels=np.linspace(0, 100, 11), cmap='viridis', extend='max') # Contour plot with filled levels

# 3. Add titles and labels
plot.ax.set_title("Pollutant Concentration Cross-Section")
plot.ax.set_xlabel("Horizontal Distance (km)")
plot.ax.set_ylabel("Vertical Height (km)")
plt.colorbar(plot.ax.collections[0], label="Concentration (µg/m³)") # Add colorbar

plt.tight_layout()
plt.show()
```

### Expected Output

A 2D contour plot will be displayed. The x-axis will represent "Horizontal Distance (km)" and the y-axis "Vertical Height (km)". Filled contours will show the distribution of "Concentration (µg/m³)" across this cross-section, with colors from the 'viridis' colormap indicating different concentration levels. A color bar will be present to interpret the concentration values.

### Key Concepts

-   **`z`**: When a `z` array is provided, `ProfilePlot` uses `matplotlib.pyplot.contourf` to create a filled contour plot.
-   **`levels`**: Defines the contour levels to be drawn.
-   **`cmap`**: Specifies the colormap for the filled contours.
-   **`extend='max'`**: Extends the colormap to indicate values beyond the highest contour level.
-   **`plt.colorbar()`**: Essential for interpreting the color-coded values in a contour plot.
