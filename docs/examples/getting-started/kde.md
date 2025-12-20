# Kernel Density Estimate (KDE) Plots

Kernel Density Estimate (KDE) plots are used to visualize the probability density function of a continuous variable. For bivariate data, a 2D KDE plot shows the joint distribution of two variables, often represented by contours or a color gradient. It's a powerful way to understand the shape and concentration of data points in a continuous space.

## Prerequisites

-   Basic Python knowledge
-   Understanding of numpy and pandas
-   MONET Plots installed (`pip install monet_plots`)
-   Seaborn installed (`pip install seaborn`)
-   Matplotlib installed (`pip install matplotlib`)

## Plotting Workflow

To create a KDE plot with `monet_plots`:

1.  **Prepare Data**: Ensure your data is in a `pandas.DataFrame`, `xarray.Dataset`, or `xarray.DataArray` format, containing the two numerical variables you wish to plot.
2.  **Initialize `KDEPlot`**: Create an instance of `KDEPlot`, passing your data, and the column names for the `x` and `y` axes.
3.  **Call `plot` method**: Generate the KDE plot.
4.  **Customize (Optional)**: Add titles, labels, or other aesthetic properties.
5.  **Display/Save**: Show the plot using `plt.show()` or save it to a file.

## Example 1: Basic 2D KDE Plot

Let's create a basic 2D KDE plot to visualize the joint distribution of two correlated numerical variables.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from monet_plots.plots.kde import KDEPlot

# 1. Prepare sample data
np.random.seed(42) # for reproducibility
mean = [0, 0]
cov = [[1, 0.8], [0.8, 1]] # Correlated variables
data = np.random.multivariate_normal(mean, cov, 1000)
df = pd.DataFrame(data, columns=['Variable A', 'Variable B'])

# 2. Initialize and create the plot
plot = KDEPlot(df=df, x='Variable A', y='Variable B', title="2D Kernel Density Estimate of Correlated Variables", figsize=(8, 7))
plot.plot() # Default KDE plot

# 3. Add titles and labels
plot.ax.set_xlabel("Variable A")
plot.ax.set_ylabel("Variable B")

plt.tight_layout()
plt.show()
```

### Expected Output

A 2D plot will be displayed with 'Variable A' on the x-axis and 'Variable B' on the y-axis. Contour lines will represent the density of data points, showing a higher concentration where the variables are more frequently observed together. Due to the positive correlation in the sample data, the contours will likely stretch diagonally from the bottom-left to the top-right.

### Key Concepts

-   **`KDEPlot`**: The class used to generate 2D Kernel Density Estimate plots.
-   **`df`**: The DataFrame containing the data.
-   **`x` and `y`**: The column names from the DataFrame to be used for the x and y axes, respectively.
-   **Contour Lines**: Represent regions of equal probability density, with denser lines indicating higher concentrations of data.

## Example 2: KDE Plot with Customization (Filled Contours and Colormap)

You can customize the appearance of the KDE plot, for instance, by filling the contours with color and choosing a specific colormap to enhance readability.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from monet_plots.plots.kde import KDEPlot

# 1. Prepare sample data (same as above)
np.random.seed(42) # for reproducibility
mean = [0, 0]
cov = [[1, 0.8], [0.8, 1]]
data = np.random.multivariate_normal(mean, cov, 1000)
df = pd.DataFrame(data, columns=['Variable A', 'Variable B'])

# 2. Initialize and create the plot with custom parameters
plot = KDEPlot(df=df, x='Variable A', y='Variable B', figsize=(8, 7))
plot.plot(
    fill=True,      # Fill the contours with color
    cmap='viridis', # Use the 'viridis' colormap
    cbar=True,      # Add a color bar
    cbar_kws={'label': 'Density'} # Label for the color bar
)

# 3. Add titles and labels
plot.ax.set_title("Filled 2D KDE Plot with Viridis Colormap")
plot.ax.set_xlabel("Variable A")
plot.ax.set_ylabel("Variable B")

plt.tight_layout()
plt.show()
```

### Expected Output

Similar to the basic KDE plot, but the areas between the contour lines will be filled with colors from the 'viridis' colormap. A color bar will be present, indicating the density values corresponding to the colors. This often provides a more intuitive visual representation of the density distribution.

### Key Concepts

-   **`fill=True`**: Fills the area between the contour lines, creating a heatmap-like appearance.
-   **`cmap='viridis'`**: Specifies the colormap to use for filling the contours. Many Matplotlib colormaps are available.
-   **`cbar=True`**: Adds a color bar to the plot, which helps in interpreting the density values represented by the colors.
-   **`cbar_kws`**: A dictionary of keyword arguments passed to the color bar creation, allowing for customization like adding a label.
