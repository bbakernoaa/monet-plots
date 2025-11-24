# Performance Diagram Plots

The Performance Diagram (also known as the Roebber Diagram) is a powerful tool for evaluating and comparing the skill of categorical forecasts. It simultaneously displays four key verification metrics: Probability of Detection (POD), Success Ratio (SR), Critical Success Index (CSI), and Bias. This allows forecasters and model developers to quickly assess the strengths and weaknesses of different forecast systems.

## Prerequisites

-   Basic Python knowledge
-   Understanding of numpy and pandas
-   MONET Plots installed (`pip install monet_plots`)
-   Matplotlib installed (`pip install matplotlib`)

## Plotting Workflow

To create a performance diagram with `monet_plots`:

1.  **Prepare Data**: Your data can be either:
    *   A `pandas.DataFrame` with pre-calculated `success_ratio` and `pod` columns.
    *   A `pandas.DataFrame` with contingency table counts (`hits`, `misses`, `false_alarms`, `correct_negatives`), from which `success_ratio` and `pod` will be computed.
2.  **Initialize `PerformanceDiagramPlot`**: Create an instance of the `PerformanceDiagramPlot` class.
3.  **Call `plot` method**: Pass your data and specify the relevant column names.
4.  **Customize (Optional)**: Add titles, labels, or other visual enhancements.
5.  **Display/Save**: Show the plot using `plt.show()` or save it to a file.

## Example 1: Basic Performance Diagram from Pre-calculated Metrics

Let's create a performance diagram using pre-calculated Probability of Detection (POD) and Success Ratio (SR) values for a single forecast system.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from monet_plots.plots.performance_diagram import PerformanceDiagramPlot

# 1. Prepare sample data with pre-calculated metrics
# Each row represents a different forecast threshold or scenario
data = {
    'success_ratio': [0.2, 0.4, 0.6, 0.8, 0.9],
    'pod': [0.3, 0.6, 0.7, 0.85, 0.92],
    'model': ['Model A'] * 5 # For a single model
}
df = pd.DataFrame(data)

# 2. Initialize and create the plot
plot = PerformanceDiagramPlot(figsize=(8, 8))
plot.plot(df, x_col='success_ratio', y_col='pod', markersize=8, color='blue', label='Forecast System')

# 3. Add titles and labels (optional, but good practice)
plot.ax.set_title("Performance Diagram for a Forecast System")
plot.ax.legend(loc='lower right') # Add legend for the plotted points

plt.tight_layout()
plt.show()
```

### Expected Output

A square plot will be displayed with "Success Ratio (1-FAR)" on the x-axis and "Probability of Detection (POD)" on the y-axis, both ranging from 0 to 1. Several dashed gray isolines for Critical Success Index (CSI) and dotted gray isolines for Bias will be drawn in the background. Blue circular markers will represent the performance of the forecast system at different thresholds, and a black 1:1 line will also be present.

### Key Concepts

-   **`PerformanceDiagramPlot`**: The class used to generate performance diagrams.
-   **`x_col='success_ratio'`**: Specifies the column containing the Success Ratio (1 - False Alarm Ratio).
-   **`y_col='pod'`**: Specifies the column containing the Probability of Detection.
-   **Isolines**: The background lines for CSI and Bias help interpret the performance of the plotted points.
-   **Perfect Forecast**: The top-right corner (SR=1, POD=1) represents a perfect forecast.

## Example 2: Performance Diagram from Contingency Table Counts

Often, you have raw contingency table counts (hits, misses, false alarms, correct negatives) rather than pre-calculated metrics. `PerformanceDiagramPlot` can compute POD and SR directly from these counts.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from monet_plots.plots.performance_diagram import PerformanceDiagramPlot

# 1. Prepare sample data with contingency table counts
# Each row represents a different forecast threshold or scenario
data_counts = {
    'hits': [10, 20, 30, 40, 45],
    'misses': [30, 20, 15, 10, 5],
    'false_alarms': [50, 30, 20, 10, 5],
    'correct_negatives': [100, 120, 130, 140, 145],
    'model': ['Model B'] * 5
}
df_counts = pd.DataFrame(data_counts)

# Define the columns for hits, misses, false alarms, and correct negatives
contingency_cols = ['hits', 'misses', 'false_alarms', 'correct_negatives']

# 2. Initialize and create the plot, specifying counts_cols
plot = PerformanceDiagramPlot(figsize=(8, 8))
plot.plot(df_counts, counts_cols=contingency_cols, markersize=8, color='green', label='Forecast System (from counts)')

plot.ax.set_title("Performance Diagram from Contingency Counts")
plot.ax.legend(loc='lower right')

plt.tight_layout()
plt.show()
```

### Expected Output

Similar to Example 1, but the POD and SR values are computed internally from the provided `hits`, `misses`, `false_alarms`, and `correct_negatives` columns. Green square markers will indicate the performance points.

### Key Concepts

-   **`counts_cols`**: A list of four strings `[hits_col, misses_col, false_alarms_col, correct_negatives_col]` that tells the plot to calculate POD and SR internally.
-   **Contingency Table**: The fundamental data structure for evaluating categorical forecasts.

## Example 3: Comparing Multiple Forecast Systems

The `label_col` parameter allows you to plot and compare the performance of multiple forecast systems or models on the same diagram.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from monet_plots.plots.performance_diagram import PerformanceDiagramPlot

# 1. Prepare sample data for two different models
# Model 1 (better performance)
data_model1 = {
    'success_ratio': [0.3, 0.5, 0.7, 0.85],
    'pod': [0.4, 0.7, 0.8, 0.9],
    'system': ['Model 1'] * 4
}
df_model1 = pd.DataFrame(data_model1)

# Model 2 (lower performance)
data_model2 = {
    'success_ratio': [0.1, 0.3, 0.5, 0.6],
    'pod': [0.2, 0.4, 0.6, 0.7],
    'system': ['Model 2'] * 4
}
df_model2 = pd.DataFrame(data_model2)

# Combine dataframes
df_combined = pd.concat([df_model1, df_model2]).reset_index(drop=True)

# 2. Initialize and create the plot, using 'system' as label_col
plot = PerformanceDiagramPlot(figsize=(8, 8))
plot.plot(df_combined, x_col='success_ratio', y_col='pod', label_col='system', markersize=8)

plot.ax.set_title("Performance Diagram: Comparing Two Forecast Systems")
# Legend is automatically added when label_col is used

plt.tight_layout()
plt.show()
```

### Expected Output

A single performance diagram will show two distinct sets of points, each representing a different forecast system ('Model 1' and 'Model 2'), distinguished by color and shape (default). A legend will automatically be generated to identify each system. This allows for a direct visual comparison of their performance characteristics.

### Key Concepts

-   **`label_col`**: Specifies a column whose unique values will be used to group the data points, plotting each group with a different color/marker and adding an automatic legend.
-   **Comparative Analysis**: This feature is essential for comparing multiple models, different configurations of a single model, or performance across various regions or time periods.