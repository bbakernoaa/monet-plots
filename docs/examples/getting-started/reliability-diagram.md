# Reliability Diagram Plots

Reliability diagrams (also known as attributes diagrams) are a fundamental tool for assessing the calibration of probabilistic forecasts. They plot the observed frequency of an event against the forecast probability of that event. A perfectly reliable forecast system will have its points fall along the 1:1 diagonal line, meaning that when the forecast predicts a 30% chance of rain, it actually rains 30% of the time. Deviations from this line indicate a lack of calibration.

## Prerequisites

-   Basic Python knowledge
-   Understanding of numpy and pandas
-   MONET Plots installed (`pip install monet_plots`)
-   Matplotlib installed (`pip install matplotlib`)

## Plotting Workflow

To create a reliability diagram with `monet_plots`:

1.  **Prepare Data**: Your data should be a `pandas.DataFrame` containing:
    *   A column of continuous forecast probabilities (between 0 and 1).
    *   A column of binary observations (0 for no event, 1 for event).
2.  **Initialize `ReliabilityDiagramPlot`**: Create an instance of the `ReliabilityDiagramPlot` class.
3.  **Call `plot` method**: Pass your data, specifying the `forecasts_col` and `observations_col`. You can also specify `n_bins` and `show_hist`.
4.  **Customize (Optional)**: Add titles, labels, or other visual enhancements.
5.  **Display/Save**: Show the plot using `plt.show()` or save it to a file.

## Example 1: Basic Reliability Diagram from Raw Forecasts and Observations

Let's create a basic reliability diagram using synthetic probabilistic forecast and observation data.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from monet_plots.plots.reliability_diagram import ReliabilityDiagramPlot

# 1. Prepare sample data
np.random.seed(42) # for reproducibility
n_samples = 1000

# Simulate forecast probabilities (e.g., probability of rain)
forecast_probabilities = np.random.rand(n_samples)

# Simulate observations based on these probabilities (binary outcome)
# Introduce some unreliability: forecasts are slightly overconfident at low probs, underconfident at high probs
observations = (np.random.rand(n_samples) < (forecast_probabilities * 0.8 + 0.1)).astype(int)

df = pd.DataFrame({
    'forecast_prob': forecast_probabilities,
    'observed_event': observations
})

# 2. Initialize and create the plot
plot = ReliabilityDiagramPlot(figsize=(8, 8))
plot.plot(
    df,
    forecasts_col='forecast_prob',
    observations_col='observed_event',
    n_bins=10,
    markersize=8,
    color='blue',
    label='Forecast System'
)

# 3. Add titles and labels
plot.ax.set_title("Reliability Diagram for a Probabilistic Forecast")
plot.ax.legend(loc='upper left')

plt.tight_layout()
plt.show()
```

### Expected Output

A square plot will be displayed with "Forecast Probability" on the x-axis and "Observed Relative Frequency" on the y-axis, both ranging from 0 to 1. A dashed black line will represent "Perfect Reliability" (the 1:1 diagonal). A gray dotted horizontal line will indicate the climatological frequency of the event. Blue circular markers will show the observed frequency for each bin of forecast probabilities. A shaded green region will indicate areas of positive Brier Skill Score.

### Key Concepts

-   **`ReliabilityDiagramPlot`**: The class used to generate reliability diagrams.
-   **`forecasts_col`**: The column containing the continuous forecast probabilities (0-1).
-   **`observations_col`**: The column containing the binary observations (0 or 1).
-   **`n_bins`**: The number of bins used to group forecast probabilities before calculating observed frequencies.
-   **Perfect Reliability Line**: The 1:1 diagonal, indicating ideal calibration.
-   **Climatology Line**: The horizontal line representing the overall observed frequency of the event.
-   **Skill Region**: The shaded area where the forecast is more skillful than a climatological forecast.

## Example 2: Reliability Diagram with Sharpness Histogram

A reliability diagram assesses calibration, but it doesn't tell you how often each forecast probability is used (sharpness). Including a sharpness histogram as an inset can provide this crucial information.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from monet_plots.plots.reliability_diagram import ReliabilityDiagramPlot

# 1. Prepare sample data (same as Example 1)
np.random.seed(42) # for reproducibility
n_samples = 1000
forecast_probabilities = np.random.rand(n_samples)
observations = (np.random.rand(n_samples) < (forecast_probabilities * 0.8 + 0.1)).astype(int)
df = pd.DataFrame({
    'forecast_prob': forecast_probabilities,
    'observed_event': observations
})

# 2. Initialize and create the plot with sharpness histogram
plot = ReliabilityDiagramPlot(figsize=(8, 8))
plot.plot(
    df,
    forecasts_col='forecast_prob',
    observations_col='observed_event',
    n_bins=10,
    show_hist=True, # Enable the sharpness histogram
    markersize=8,
    color='red',
    label='Forecast System'
)

plot.ax.set_title("Reliability Diagram with Sharpness Histogram")
plot.ax.legend(loc='upper left')

plt.tight_layout()
plt.show()
```

### Expected Output

The main plot will be a reliability diagram similar to Example 1. Additionally, a small inset histogram will appear, typically in the upper right corner. This histogram will show the frequency of usage for each forecast probability bin, indicating the "sharpness" of the forecast system (i.e., how often it issues confident forecasts near 0 or 1).

### Key Concepts

-   **`show_hist=True`**: This argument adds an inset histogram showing the distribution of forecast probabilities.
-   **Sharpness**: Refers to the ability of a forecast system to issue confident forecasts (probabilities close to 0 or 1). A sharp forecast uses the full range of probabilities.

## Example 3: Comparing Multiple Forecast Systems

You can compare the reliability of multiple forecast systems by using the `label_col` parameter.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from monet_plots.plots.reliability_diagram import ReliabilityDiagramPlot

# 1. Prepare sample data for two different models
np.random.seed(42) # for reproducibility
n_samples = 500

# Model 1: Slightly overconfident
forecast_prob_1 = np.random.rand(n_samples)
obs_1 = (np.random.rand(n_samples) < (forecast_prob_1 * 0.9 + 0.05)).astype(int)
df_model1 = pd.DataFrame({
    'forecast_prob': forecast_prob_1,
    'observed_event': obs_1,
    'model': 'Model 1'
})

# Model 2: Slightly underconfident
forecast_prob_2 = np.random.rand(n_samples)
obs_2 = (np.random.rand(n_samples) < (forecast_prob_2 * 1.1 - 0.05)).astype(int)
df_model2 = pd.DataFrame({
    'forecast_prob': forecast_prob_2,
    'observed_event': obs_2,
    'model': 'Model 2'
})

# Combine dataframes
df_combined = pd.concat([df_model1, df_model2]).reset_index(drop=True)

# 2. Initialize and create the plot, using 'model' as label_col
plot = ReliabilityDiagramPlot(figsize=(8, 8))
plot.plot(
    df_combined,
    forecasts_col='forecast_prob',
    observations_col='observed_event',
    n_bins=10,
    label_col='model',
    markersize=8
)

plot.ax.set_title("Reliability Diagram: Comparing Two Forecast Systems")
# Legend is automatically added when label_col is used

plt.tight_layout()
plt.show()
```

### Expected Output

A single reliability diagram will show two sets of points, one for 'Model 1' and one for 'Model 2', distinguished by color and an automatic legend. You will be able to visually compare their calibration curves against the perfect reliability line and the climatology line, identifying which model is better calibrated.

### Key Concepts

-   **`label_col`**: Allows plotting multiple reliability curves on the same axes, grouped by a specified categorical column.
-   **Comparative Analysis**: This feature is essential for comparing the calibration of different probabilistic forecast systems or different configurations of a single system.
