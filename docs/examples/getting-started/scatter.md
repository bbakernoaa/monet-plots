# Scatter Plots

Scatter plots are a fundamental visualization tool used to display the relationship between two numerical variables. Each point on the plot represents an observation, with its position determined by the values of the two variables. MONET Plots' `ScatterPlot` class extends this by integrating `seaborn.regplot`, allowing for the easy addition of linear regression lines and confidence intervals to highlight trends.

## Prerequisites

-   Basic Python knowledge
-   Understanding of numpy and pandas
-   MONET Plots installed (`pip install monet_plots`)
-   Seaborn installed (`pip install seaborn`)
-   Matplotlib installed (`pip install matplotlib`)

## Plotting Workflow

To create a scatter plot with `monet_plots`:

1.  **Prepare Data**: Ensure your data is in a `pandas.DataFrame`, `xarray.Dataset`, or `xarray.DataArray` format, containing the numerical variables you wish to plot.
2.  **Initialize `ScatterPlot`**: Create an instance of `ScatterPlot`, passing your data, and the column names for the `x` and `y` axes.
3.  **Call `plot` method**: Generate the scatter plot, optionally passing additional keyword arguments for customization.
4.  **Customize (Optional)**: Add titles, labels, or other aesthetic properties.
5.  **Display/Save**: Show the plot using `plt.show()` or save it to a file.

## Example 1: Basic Scatter Plot with Regression Line

Let's create a basic scatter plot to visualize the relationship between two simulated variables, along with a linear regression line.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from monet_plots.plots.scatter import ScatterPlot

# 1. Prepare sample data
np.random.seed(42) # for reproducibility
n_samples = 100

# Simulate two correlated variables
x_data = np.random.normal(loc=10, scale=2, size=n_samples)
y_data = 0.7 * x_data + np.random.normal(loc=5, scale=1.5, size=n_samples)

df = pd.DataFrame({
    'predictor': x_data,
    'response': y_data
})

# 2. Initialize and create the plot
plot = ScatterPlot(df=df, x='predictor', y='response', title="Scatter Plot of Response vs. Predictor", figsize=(9, 7))
plot.plot(
    scatter_kws={'alpha': 0.7, 's': 60, 'edgecolor': 'w'}, # kwargs for scatter points
    line_kws={'color': 'red', 'linewidth': 2} # kwargs for regression line
)

# 3. Add titles and labels
plot.ax.set_xlabel("Predictor Variable")
plot.ax.set_ylabel("Response Variable")
plot.ax.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
```

### Expected Output

A scatter plot will be displayed with "Predictor Variable" on the x-axis and "Response Variable" on the y-axis. Individual data points will be shown as blue circles. A red linear regression line will be drawn through the points, accompanied by a light blue shaded area representing the 95% confidence interval for the regression estimate.

### Key Concepts

-   **`ScatterPlot`**: The class used to generate scatter plots with optional regression.
-   **`df`**: The DataFrame containing the data.
-   **`x` and `y`**: The column names from the DataFrame to be used for the x and y axes, respectively.
-   **Regression Line**: Automatically fitted and displayed by `seaborn.regplot`, showing the linear trend between the variables.
-   **Confidence Interval**: The shaded area around the regression line, indicating the uncertainty of the regression estimate.

## Example 2: Scatter Plot with Multiple Y-variables

You can plot multiple response variables against a single predictor variable on the same scatter plot.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from monet_plots.plots.scatter import ScatterPlot

# 1. Prepare sample data with multiple response variables
np.random.seed(42) # for reproducibility
n_samples = 100

x_data = np.random.normal(loc=10, scale=2, size=n_samples)
y1_data = 0.7 * x_data + np.random.normal(loc=5, scale=1.5, size=n_samples)
y2_data = -0.5 * x_data + np.random.normal(loc=20, scale=2, size=n_samples)

df_multi_y = pd.DataFrame({
    'predictor': x_data,
    'response_1': y1_data,
    'response_2': y2_data
})

# 2. Initialize and create the plot with a list of y-variables
plot = ScatterPlot(df=df_multi_y, x='predictor', y=['response_1', 'response_2'], title="Scatter Plot of Multiple Responses vs. Predictor", figsize=(9, 7))
plot.plot(
    scatter_kws={'alpha': 0.6, 's': 50, 'edgecolor': 'w'},
    line_kws={'linewidth': 2}
)

# 3. Add titles and labels
plot.ax.set_xlabel("Predictor Variable")
plot.ax.set_ylabel("Response Variable Value")
plot.ax.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
```

### Expected Output

A single scatter plot will display two sets of points and two regression lines. One set will represent 'response\_1' vs. 'predictor' (e.g., in blue), and the other 'response\_2' vs. 'predictor' (e.g., in orange). Each will have its own regression line and confidence interval. A legend will differentiate between 'response\_1' and 'response\_2'.

### Key Concepts

-   **`y=['response_1', 'response_2']`**: Passing a list of column names to the `y` parameter allows plotting multiple relationships on the same axes.
-   **Automatic Legend**: When multiple `y` variables are provided, a legend is automatically generated to distinguish them.

## Example 3: Customizing Regression Plot Appearance

You can pass a wide range of keyword arguments directly to the `plot` method, which are then forwarded to `seaborn.regplot`, allowing for extensive customization of both the scatter points and the regression line.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from monet_plots.plots.scatter import ScatterPlot

# 1. Prepare sample data (same as Example 1)
np.random.seed(42) # for reproducibility
n_samples = 100
x_data = np.random.normal(loc=10, scale=2, size=n_samples)
y_data = 0.7 * x_data + np.random.normal(loc=5, scale=1.5, size=n_samples)
df = pd.DataFrame({
    'predictor': x_data,
    'response': y_data
})

# 2. Initialize and create the plot with extensive customization
plot = ScatterPlot(df=df, x='predictor', y='response', title="Customized Scatter Plot with Regression", figsize=(9, 7))
plot.plot(
    color='purple', # Color for scatter points and regression line
    marker='X',     # Marker style for scatter points
    s=100,          # Size of scatter points
    alpha=0.8,      # Transparency of scatter points
    ci=68,          # 68% confidence interval (instead of default 95%)
    line_kws={
        'linestyle': '--', # Dashed regression line
        'color': 'darkgreen', # Different color for the line
        'linewidth': 3
    },
    scatter_kws={
        'edgecolor': 'black', # Black edge for markers
        'facecolor': 'lightgray' # Light gray fill for markers
    }
)

# 3. Add titles and labels
plot.ax.set_xlabel("Predictor Variable")
plot.ax.set_ylabel("Response Variable")
plot.ax.grid(True, linestyle=':', alpha=0.7)

plt.tight_layout()
plt.show()
```

### Expected Output

A scatter plot with highly customized appearance. The scatter points will be large 'X' markers, filled with light gray and a black edge. The regression line will be a thick, dashed dark green line, and the confidence interval will be shaded for a 68% confidence level.

### Key Concepts

-   **`scatter_kws` and `line_kws`**: Dictionaries that allow passing specific keyword arguments directly to the underlying Matplotlib scatter plot and line plot functions, respectively, giving fine-grained control over their appearance.
-   **`ci`**: Controls the size of the confidence interval around the regression line. Set to `None` to remove the confidence interval.
-   **Direct `kwargs`**: Many `seaborn.regplot` parameters (like `color`, `marker`, `s`, `alpha`) can be passed directly to `plot.plot()`.