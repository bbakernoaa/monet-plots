# Relative Economic Value (REV) Plots

Relative Economic Value (REV) plots are a powerful tool for assessing the economic utility of a probabilistic forecast system. They quantify the potential savings or benefits a user can gain by using a forecast compared to a "climatology" (no-skill) forecast, across a range of cost/loss ratios. A higher REV indicates a more economically valuable forecast.

## Prerequisites

-   Basic Python knowledge
-   Understanding of numpy and pandas
-   MONET Plots installed (`pip install monet_plots`)
-   Matplotlib installed (`pip install matplotlib`)

## Plotting Workflow

To create an REV plot with `monet_plots`:

1.  **Prepare Data**: Your data should be a `pandas.DataFrame` containing contingency table counts: `hits`, `misses`, `false_alarms` (fa), and `correct_negatives` (cn).
2.  **Initialize `RelativeEconomicValuePlot`**: Create an instance of the `RelativeEconomicValuePlot` class.
3.  **Call `plot` method**: Pass your data, specifying the `counts_cols`. You can also provide a `climatology` or `cost_loss_ratios`.
4.  **Customize (Optional)**: Add titles, labels, or other visual enhancements.
5.  **Display/Save**: Show the plot using `plt.show()` or save it to a file.

## Example 1: Basic REV Plot for a Single Forecast System

Let's create a basic REV plot for a single forecast system using synthetic contingency table counts.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from monet_plots.plots.rev import RelativeEconomicValuePlot

# 1. Prepare sample data with contingency table counts
# These counts represent the performance of a forecast system over many events
data = {
    'hits': [100],
    'misses': [20],
    'fa': [30], # False Alarms
    'cn': [850] # Correct Negatives
}
df = pd.DataFrame(data)

# Define the columns for hits, misses, false alarms, and correct negatives
contingency_cols = ['hits', 'misses', 'fa', 'cn']

# 2. Initialize and create the plot
plot = RelativeEconomicValuePlot(figsize=(10, 7))
plot.plot(
    df,
    counts_cols=contingency_cols,
    color='blue',
    linewidth=2
)

# 3. Add titles and labels
plot.ax.set_title("Relative Economic Value of a Forecast System")
plot.ax.legend(loc='lower right')

plt.tight_layout()
plt.show()
```

### Expected Output

A plot will be displayed with "Cost/Loss Ratio" on the x-axis (ranging from 0 to 1) and "Relative Economic Value (REV)" on the y-axis (ranging from -0.2 to 1.05). A blue line will represent the REV curve for the forecast system. A dashed black line at REV=0 (representing climatology) and a dotted gray line at REV=1 (representing a perfect forecast) will also be present. The REV curve will show how the economic value changes with different cost/loss ratios.

### Key Concepts

-   **`RelativeEconomicValuePlot`**: The class used to generate REV plots.
-   **`counts_cols`**: A list of four strings `[hits_col, misses_col, fa_col, cn_col]` that provides the contingency table counts.
-   **Cost/Loss Ratio (C/L)**: The ratio of the cost of taking protective action (C) to the loss incurred if no action is taken and the event occurs (L). It's a critical parameter for decision-making.
-   **REV = 0 Line**: Represents the economic value of using a climatological forecast (no skill).
-   **REV = 1 Line**: Represents the economic value of a perfect forecast.

## Example 2: Comparing Multiple Forecast Systems with REV

The `label_col` parameter allows you to compare the economic value of multiple forecast systems on the same plot, helping users choose the most beneficial forecast for their specific cost/loss scenario.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from monet_plots.plots.rev import RelativeEconomicValuePlot

# 1. Prepare sample data for two different models
# Model 1: Higher overall skill
data_model1 = {
    'hits': [100], 'misses': [20], 'fa': [30], 'cn': [850], 'model': ['Model 1']
}
df_model1 = pd.DataFrame(data_model1)

# Model 2: Lower overall skill or different performance characteristics
data_model2 = {
    'hits': [80], 'misses': [40], 'fa': [50], 'cn': [830], 'model': ['Model 2']
}
df_model2 = pd.DataFrame(data_model2)

# Combine dataframes
df_combined = pd.concat([df_model1, df_model2]).reset_index(drop=True)

# Define the columns for hits, misses, false alarms, and correct negatives
contingency_cols = ['hits', 'misses', 'fa', 'cn']

# 2. Initialize and create the plot, using 'model' as label_col
plot = RelativeEconomicValuePlot(figsize=(10, 7))
plot.plot(
    df_combined,
    counts_cols=contingency_cols,
    label_col='model',
    linewidth=2
)

plot.ax.set_title("Relative Economic Value: Comparing Two Forecast Systems")
# Legend is automatically added when label_col is used

plt.tight_layout()
plt.show()
```

### Expected Output

A single plot will display two REV curves, one for 'Model 1' and one for 'Model 2', distinguished by color and an automatic legend. This allows for a direct visual comparison of their economic value across the range of cost/loss ratios. You can identify which model provides greater economic benefit for different C/L thresholds.

### Key Concepts

-   **`label_col`**: Allows plotting multiple REV curves on the same axes, grouped by a specified categorical column (e.g., different models).
-   **Decision Making**: REV plots are crucial for decision-makers to understand which forecast system is most beneficial given their specific cost/loss considerations.
