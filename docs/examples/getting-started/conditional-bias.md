# Conditional Bias Plots

Conditional bias plots are used to visualize the bias (Forecast - Observation) as a function of the observed value. This type of plot helps in understanding how the model's bias changes across different ranges of observed values, often revealing systematic errors or dependencies. It's particularly useful in model verification to diagnose where a model performs well or poorly.

## Prerequisites

-   Basic Python knowledge
-   Understanding of numpy and pandas
-   MONET Plots installed (`pip install monet_plots`)
-   Matplotlib installed (`pip install matplotlib`)

## Plotting Workflow

The workflow for creating a conditional bias plot involves:

1.  **Prepare Data**: Organize your paired forecast and observation data into a `pandas.DataFrame`, `xarray.Dataset`, or `xarray.DataArray`.
2.  **Initialize `ConditionalBiasPlot`**: Create an instance of the `ConditionalBiasPlot` class.
3.  **Call `plot` method**: Pass your data, specifying the observation and forecast column names, and optionally the number of bins.
4.  **Customize (Optional)**: Add titles, labels, and other visual enhancements.
5.  **Display/Save**: Show the plot using `plt.show()` or save it to a file.

## Example 1: Basic Conditional Bias Plot

Let's create a conditional bias plot using synthetic observation and forecast data to illustrate how bias can vary with observed values.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from monet_plots.plots.conditional_bias import ConditionalBiasPlot

# 1. Prepare sample data
np.random.seed(42) # for reproducibility
n_samples = 500

# Simulate observations
observations = np.random.normal(loc=10, scale=3, size=n_samples)

# Simulate forecasts with a bias that depends on the observation value:
# - For lower observations, forecast is slightly higher (positive bias)
# - For higher observations, forecast is slightly lower (negative bias)
# - Add some random noise
forecasts = observations + (0.5 - 0.1 * observations) + np.random.normal(loc=0, scale=0.5, size=n_samples)

# Ensure data is in a DataFrame
df = pd.DataFrame({'observations': observations, 'forecasts': forecasts})

# 2. Initialize and create the plot
plot = ConditionalBiasPlot(figsize=(10, 6))
plot.plot(df, obs_col='observations', fcst_col='forecasts', n_bins=15)

# 3. Add titles and labels
plot.ax.set_title("Conditional Bias Plot (Forecast vs. Observation)")
plot.ax.set_xlabel("Observed Value")
plot.ax.set_ylabel("Mean Bias (Forecast - Observation)")

plt.tight_layout()
plt.show()
```

### Expected Output

A plot will be displayed showing the mean bias (Forecast - Observation) on the y-axis against the observed value on the x-axis. The x-axis will be divided into 15 bins, and for each bin, a point will represent the mean bias, with vertical error bars indicating the standard deviation of the bias within that bin. A horizontal dashed line at y=0 will represent zero bias. You should observe a trend where the bias changes from positive to negative as observed values increase, indicating a conditional bias.

### Key Concepts

-   **`ConditionalBiasPlot`**: The class responsible for generating this type of plot.
-   **`obs_col`**: Specifies the name of the column in your data that contains the observed values.
-   **`fcst_col`**: Specifies the name of the column in your data that contains the forecast values.
-   **`n_bins`**: Controls the number of bins used to discretize the observed values. A higher number of bins provides more detail but may result in fewer samples per bin.
-   **Zero Bias Line**: The horizontal dashed line at y=0 serves as a reference, indicating where the forecast perfectly matches the observation on average. Deviations from this line highlight the presence and magnitude of bias.
-   **Error Bars**: The vertical error bars typically represent the standard deviation of the bias within each bin, giving an indication of the variability or uncertainty of the bias estimate.

## Example 2: Conditional Bias Plot with Grouping

You can also analyze conditional bias for different groups within your data by specifying a `label_col`. This is useful for comparing the performance of different models or different categories of data.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from monet_plots.plots.conditional_bias import ConditionalBiasPlot

# 1. Prepare sample data with a grouping column
np.random.seed(42) # for reproducibility
n_samples_per_group = 250

# Group 1: Model A (slight positive bias)
obs_A = np.random.normal(loc=10, scale=3, size=n_samples_per_group)
fcst_A = obs_A + 1.0 + np.random.normal(loc=0, scale=0.5, size=n_samples_per_group)
df_A = pd.DataFrame({'observations': obs_A, 'forecasts': fcst_A, 'model': 'Model A'})

# Group 2: Model B (slight negative bias)
obs_B = np.random.normal(loc=12, scale=2, size=n_samples_per_group)
fcst_B = obs_B - 0.5 + np.random.normal(loc=0, scale=0.4, size=n_samples_per_group)
df_B = pd.DataFrame({'observations': obs_B, 'forecasts': fcst_B, 'model': 'Model B'})

# Combine dataframes
df_combined = pd.concat([df_A, df_B]).reset_index(drop=True)

# 2. Initialize and create the plot with grouping
plot = ConditionalBiasPlot(figsize=(12, 7))
plot.plot(df_combined, obs_col='observations', fcst_col='forecasts', n_bins=10, label_col='model')

# 3. Add titles and labels
plot.ax.set_title("Conditional Bias Plot by Model")
plot.ax.set_xlabel("Observed Value")
plot.ax.set_ylabel("Mean Bias (Forecast - Observation)")

plt.tight_layout()
plt.show()
```

### Expected Output

A single plot will be displayed, containing two sets of conditional bias lines, one for 'Model A' and one for 'Model B', each distinguished by a different color and a legend. This allows for a direct comparison of how the conditional bias differs between the two models across the range of observed values.

### Key Concepts

-   **`label_col`**: This parameter allows you to specify a column in your DataFrame whose unique values will be used to group the data, plotting a separate conditional bias line for each group. This is invaluable for comparative analysis.
-   **Comparative Analysis**: By plotting multiple groups on the same axes, you can easily identify which models or categories exhibit more or less bias, and how that bias changes conditionally.
