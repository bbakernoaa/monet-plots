# Scorecard Plots

Scorecard plots provide a concise, visual summary of performance metrics across multiple dimensions, often presented as a heatmap. They are particularly useful for comparing the performance of different models, forecast lead times, or variables at a glance. By coloring cells based on a metric (e.g., difference from a baseline) and optionally adding significance markers, scorecards quickly highlight areas of strength and weakness.

## Prerequisites

-   Basic Python knowledge
-   Understanding of numpy and pandas
-   MONET Plots installed (`pip install monet_plots`)
-   Seaborn installed (`pip install seaborn`)
-   Matplotlib installed (`pip install matplotlib`)

## Plotting Workflow

To create a scorecard plot with `monet_plots`:

1.  **Prepare Data**: Your data should be in a "long-format" `pandas.DataFrame` with columns representing the two dimensions for the grid (e.g., 'variable', 'lead\_time'), a column for the metric value to be colored, and optionally a column for significance.
2.  **Initialize `ScorecardPlot`**: Create an instance of the `ScorecardPlot` class.
3.  **Call `plot` method**: Pass your data, specifying the `x_col`, `y_col`, `val_col`, and optionally `sig_col`.
4.  **Customize (Optional)**: Adjust the colormap, center value, or other heatmap parameters.
5.  **Display/Save**: Show the plot using `plt.show()` or save it to a file.

## Example 1: Basic Scorecard Plot

Let's create a basic scorecard plot to visualize how a performance metric (e.g., RMSE difference from a control run) varies across different variables and forecast lead times.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from monet_plots.plots.scorecard import ScorecardPlot

# 1. Prepare sample data in long format
np.random.seed(42) # for reproducibility

variables = ['Temperature', 'Humidity', 'Wind Speed', 'Pressure']
lead_times = ['+06h', '+12h', '+18h', '+24h', '+36h']

data_list = []
for var in variables:
    for lt in lead_times:
        # Simulate a metric value (e.g., RMSE difference from baseline)
        # Values around 0 mean similar performance, positive means worse, negative means better
        metric_value = np.random.normal(loc=0, scale=0.5)
        if var == 'Temperature' and lt == '+06h':
            metric_value = -1.2 # Example of good performance
        elif var == 'Pressure' and lt == '+36h':
            metric_value = 1.5 # Example of poor performance

        data_list.append({
            'Variable': var,
            'Lead Time': lt,
            'Metric Value': metric_value
        })

df = pd.DataFrame(data_list)

# 2. Initialize and create the plot
plot = ScorecardPlot(figsize=(10, 7))
plot.plot(
    df,
    x_col='Lead Time',
    y_col='Variable',
    val_col='Metric Value',
    cmap='RdYlGn_r', # Red-Yellow-Green colormap, reversed so green is good (negative values)
    center=0,       # Center the colormap at 0
    linewidths=.5,  # Add lines between cells
    linecolor='black'
)

# 3. Add titles and labels (plot.plot sets default title and labels)
plot.ax.set_title("Model Performance Scorecard (RMSE Difference)", fontsize=14)
plot.ax.set_xlabel("Forecast Lead Time")
plot.ax.set_ylabel("Meteorological Variable")

plt.tight_layout()
plt.show()
```

### Expected Output

A heatmap will be displayed with "Lead Time" on the x-axis and "Variable" on the y-axis. Each cell will be colored according to its 'Metric Value', with a diverging colormap (e.g., green for negative values indicating better performance, red for positive values indicating worse performance, and yellow/white around zero). The numerical 'Metric Value' will be annotated in each cell.

### Key Concepts

-   **`ScorecardPlot`**: The class used to generate heatmap-style scorecards.
-   **`x_col`, `y_col`**: Columns defining the rows and columns of the heatmap grid.
-   **`val_col`**: The column containing the numerical values that determine the cell colors.
-   **`cmap='RdYlGn_r'`**: A diverging colormap (Red-Yellow-Green) is often suitable for performance metrics where a central value (e.g., zero difference) is significant. `_r` reverses the colormap.
-   **`center=0`**: Ensures that the colormap's neutral color (e.g., yellow) is mapped to the value 0, making it easy to distinguish positive from negative performance.

## Example 2: Scorecard Plot with Significance Markers

You can enhance the scorecard by adding markers to cells where the performance difference is statistically significant.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from monet_plots.plots.scorecard import ScorecardPlot

# 1. Prepare sample data including a significance column
np.random.seed(42) # for reproducibility

variables = ['Temperature', 'Humidity', 'Wind Speed', 'Pressure']
lead_times = ['+06h', '+12h', '+18h', '+24h', '+36h']

data_list = []
for var in variables:
    for lt in lead_times:
        metric_value = np.random.normal(loc=0, scale=0.5)
        is_significant = np.random.rand() < 0.3 # Randomly assign significance

        if var == 'Temperature' and lt == '+06h':
            metric_value = -1.2
            is_significant = True
        elif var == 'Pressure' and lt == '+36h':
            metric_value = 1.5
            is_significant = True
        elif var == 'Humidity' and lt == '+12h':
            metric_value = -0.8
            is_significant = True

        data_list.append({
            'Variable': var,
            'Lead Time': lt,
            'Metric Value': metric_value,
            'Significant': is_significant
        })

df_sig = pd.DataFrame(data_list)

# 2. Initialize and create the plot with significance markers
plot = ScorecardPlot(figsize=(10, 7))
plot.plot(
    df_sig,
    x_col='Lead Time',
    y_col='Variable',
    val_col='Metric Value',
    sig_col='Significant', # Specify the significance column
    cmap='RdYlGn_r',
    center=0,
    linewidths=.5,
    linecolor='black'
)

plot.ax.set_title("Model Performance Scorecard with Significance", fontsize=14)
plot.ax.set_xlabel("Forecast Lead Time")
plot.ax.set_ylabel("Meteorological Variable")

plt.tight_layout()
plt.show()
```

### Expected Output

A heatmap similar to Example 1, but with an asterisk (`*`) overlaid in the center of any cell where the 'Significant' column is `True`. This visually highlights which performance differences are statistically robust, helping to focus attention on meaningful results.

### Key Concepts

-   **`sig_col`**: A column (typically boolean) that indicates whether the metric value in a cell is statistically significant. If `True`, an asterisk is overlaid.
-   **Statistical Significance**: Adding significance markers helps prevent over-interpretation of small, non-significant differences in performance.