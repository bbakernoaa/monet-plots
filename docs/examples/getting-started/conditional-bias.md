# Conditional Bias Plots

Conditional bias plots are used to visualize the bias (Forecast - Observation) as a function of the observed value. This type of plot helps in understanding how the model's bias changes across different ranges of observed values, often revealing systematic errors.

## Prerequisites

-   Basic Python knowledge
-   Understanding of numpy and pandas
-   MONET Plots installed (`pip install monet_plots`)

## Example 1: Basic Conditional Bias Plot

Let's create a conditional bias plot using synthetic observation and forecast data.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from monet_plots.plots.conditional_bias import ConditionalBiasPlot

# 1. Prepare sample data
np.random.seed(42)
n_samples = 500

# Simulate observations
observations = np.random.normal(loc=10, scale=3, size=n_samples)

# Simulate forecasts with some bias that depends on the observation value
# For lower observations, forecast is slightly higher (positive bias)
# For higher observations, forecast is slightly lower (negative bias)
forecasts = observations + (0.5 - 0.1 * observations) + np.random.normal(loc=0, scale=0.5, size=n_samples)

df = pd.DataFrame({'observations': observations, 'forecasts': forecasts})

# 2. Initialize and create the plot
plot = ConditionalBiasPlot(figsize=(10, 6))
plot.plot(df, obs_col='observations', fcst_col='forecasts', n_bins=15)

# 3. Add titles and labels (optional)
plot.title("Conditional Bias Plot (Forecast vs. Observation)")
plot.xlabel("Observed Value")
plot.ylabel("Mean Bias (Forecast - Observation)")

plt.tight_layout()
plt.show()
```

### Expected Output

A scatter plot where the x-axis represents binned observed values and the y-axis represents the mean bias (forecast - observation) within each bin. Error bars will indicate the spread of bias within each bin, and a horizontal dashed line will mark zero bias. You will observe how the bias varies with the observed values.

### Key Concepts

-   **`ConditionalBiasPlot`**: The class used to generate conditional bias plots.
-   **`obs_col`**: Specifies the column in the DataFrame containing observation data.
-   **`fcst_col`**: Specifies the column in the DataFrame containing forecast data.
-   **`n_bins`**: Determines the number of bins used to categorize the observed values, affecting the resolution of the bias analysis.
-   **Zero Bias Line**: A horizontal line at y=0, indicating perfect agreement (no bias).