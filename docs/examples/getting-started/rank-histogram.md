# Rank Histogram Plots

Rank histograms, also known as Talagrand diagrams, are a diagnostic tool used to assess the statistical consistency and reliability of ensemble forecasts. They visualize the distribution of the rank of the verifying observation within the ensemble forecast members. For a perfectly reliable ensemble, the observation should fall into each rank bin with equal probability, resulting in a flat (uniform) histogram. Deviations from flatness indicate issues with the ensemble's spread or bias.

## Prerequisites

-   Basic Python knowledge
-   Understanding of numpy and pandas
-   MONET Plots installed (`pip install monet_plots`)
-   Matplotlib installed (`pip install matplotlib`)

## Plotting Workflow

To create a rank histogram with `monet_plots`:

1.  **Prepare Data**: Your data should be a `pandas.DataFrame` containing a column of observation ranks. These ranks typically range from 0 to `n_members`, where `n_members` is the number of ensemble members.
2.  **Initialize `RankHistogramPlot`**: Create an instance of the `RankHistogramPlot` class.
3.  **Call `plot` method**: Pass your data, specifying the `rank_col` and `n_members`. You can also choose to `normalize` the frequencies.
4.  **Customize (Optional)**: Add titles, labels, or other visual enhancements.
5.  **Display/Save**: Show the plot using `plt.show()` or save it to a file.

## Example 1: Basic Rank Histogram (Uniform Distribution)

Let's simulate a perfectly reliable ensemble where the observation's rank is uniformly distributed, and plot its rank histogram.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from monet_plots.plots.rank_histogram import RankHistogramPlot

# 1. Prepare sample data for a uniform rank distribution
np.random.seed(42) # for reproducibility
n_ensemble_members = 10
n_forecasts = 1000

# Simulate ranks from 0 to n_ensemble_members (inclusive)
# A uniform distribution means each rank is equally likely
ranks_uniform = np.random.randint(0, n_ensemble_members + 1, n_forecasts)
df_uniform = pd.DataFrame({'rank': ranks_uniform})

# 2. Initialize and create the plot
plot = RankHistogramPlot(figsize=(10, 6))
plot.plot(df_uniform, rank_col='rank', n_members=n_ensemble_members, normalize=True)

# 3. Add titles and labels
plot.ax.set_title("Rank Histogram: Uniform Distribution (Reliable Ensemble)")
plot.ax.set_xlabel("Rank of Observation")
plot.ax.set_ylabel("Relative Frequency")

plt.tight_layout()
plt.show()
```

### Expected Output

A bar chart will be displayed where the x-axis represents the ranks (0 to 10) and the y-axis represents the relative frequency. All bars should be approximately the same height, forming a flat histogram. A horizontal dashed line will indicate the expected uniform frequency. This indicates a statistically reliable ensemble.

### Key Concepts

-   **`RankHistogramPlot`**: The class used to generate rank histograms.
-   **`rank_col`**: The name of the column in your DataFrame containing the observation ranks.
-   **`n_members`**: The number of ensemble members. This defines the number of bins (`n_members + 1`).
-   **`normalize=True`**: Plots relative frequencies, making it easier to compare with the theoretical uniform distribution.
-   **Flat Histogram**: The ideal shape, indicating that the ensemble is statistically consistent and its spread is appropriate.

## Example 2: Rank Histogram (Underdispersed Ensemble - U-shape)

An underdispersed ensemble has too little spread, meaning the observations frequently fall outside the range of the ensemble members. This results in a U-shaped rank histogram.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from monet_plots.plots.rank_histogram import RankHistogramPlot

# 1. Prepare sample data for an underdispersed ensemble
np.random.seed(42) # for reproducibility
n_ensemble_members = 10
n_forecasts = 1000

# Simulate ensemble members (e.g., 10 members)
ensemble_mean = np.random.normal(0, 1, n_forecasts)
ensemble_std = 0.5 # Small spread
ensemble_members = np.array([ensemble_mean + np.random.normal(0, ensemble_std, n_forecasts)
                             for _ in range(n_ensemble_members)]).T

# Simulate observations that are often outside the ensemble range
observations = ensemble_mean + np.random.normal(0, 1.5, n_forecasts) # Larger observation spread

# Calculate ranks: count how many ensemble members are less than the observation
ranks_underdispersed = np.sum(ensemble_members < observations[:, np.newaxis], axis=1)
df_underdispersed = pd.DataFrame({'rank': ranks_underdispersed})

# 2. Initialize and create the plot
plot = RankHistogramPlot(figsize=(10, 6))
plot.plot(df_underdispersed, rank_col='rank', n_members=n_ensemble_members, normalize=True, color='orange')

# 3. Add titles and labels
plot.ax.set_title("Rank Histogram: Underdispersed Ensemble (U-shape)")
plot.ax.set_xlabel("Rank of Observation")
plot.ax.set_ylabel("Relative Frequency")

plt.tight_layout()
plt.show()
```

### Expected Output

A bar chart with a U-shaped distribution will be displayed. The bars at the extreme ends (ranks 0 and `n_members`) will be significantly higher than the bars in the middle. This indicates that the observation frequently falls below the lowest ensemble member or above the highest ensemble member, suggesting the ensemble's spread is too small.

### Key Concepts

-   **U-shape**: Characteristic of an underdispersed ensemble, where the ensemble spread is too narrow to capture the true uncertainty.

## Example 3: Rank Histogram (Overdispersed Ensemble - A-shape)

An overdispersed ensemble has too much spread, meaning the observations tend to fall within the middle ranks of the ensemble members. This results in an A-shaped (or dome-shaped) rank histogram.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from monet_plots.plots.rank_histogram import RankHistogramPlot

# 1. Prepare sample data for an overdispersed ensemble
np.random.seed(42) # for reproducibility
n_ensemble_members = 10
n_forecasts = 1000

# Simulate ensemble members
ensemble_mean = np.random.normal(0, 1, n_forecasts)
ensemble_std = 1.5 # Large spread
ensemble_members = np.array([ensemble_mean + np.random.normal(0, ensemble_std, n_forecasts)
                             for _ in range(n_ensemble_members)]).T

# Simulate observations that are often within the ensemble range
observations = ensemble_mean + np.random.normal(0, 0.5, n_forecasts) # Smaller observation spread

# Calculate ranks
ranks_overdispersed = np.sum(ensemble_members < observations[:, np.newaxis], axis=1)
df_overdispersed = pd.DataFrame({'rank': ranks_overdispersed})

# 2. Initialize and create the plot
plot = RankHistogramPlot(figsize=(10, 6))
plot.plot(df_overdispersed, rank_col='rank', n_members=n_ensemble_members, normalize=True, color='green')

# 3. Add titles and labels
plot.ax.set_title("Rank Histogram: Overdispersed Ensemble (A-shape)")
plot.ax.set_xlabel("Rank of Observation")
plot.ax.set_ylabel("Relative Frequency")

plt.tight_layout()
plt.show()
```

### Expected Output

A bar chart with an A-shaped (or dome-shaped) distribution will be displayed. The bars in the middle ranks will be significantly higher than the bars at the extreme ends. This indicates that the ensemble's spread is too wide, and the observation tends to fall too frequently within the central part of the ensemble distribution.

### Key Concepts

-   **A-shape**: Characteristic of an overdispersed ensemble, where the ensemble spread is too wide, leading to observations being consistently within the ensemble range.

## Example 4: Rank Histogram with Multiple Groups

You can also compare rank histograms for different groups (e.g., different forecast models or lead times) by using the `label_col` parameter.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from monet_plots.plots.rank_histogram import RankHistogramPlot

# 1. Prepare sample data for two groups (e.g., two different models)
np.random.seed(42) # for reproducibility
n_ensemble_members = 10
n_forecasts = 500

# Group 1: Uniform ranks (reliable)
ranks_group1 = np.random.randint(0, n_ensemble_members + 1, n_forecasts)
df_group1 = pd.DataFrame({'rank': ranks_group1, 'group': 'Model A'})

# Group 2: Underdispersed ranks (U-shape)
ensemble_mean_g2 = np.random.normal(0, 1, n_forecasts)
ensemble_std_g2 = 0.5
ensemble_members_g2 = np.array([ensemble_mean_g2 + np.random.normal(0, ensemble_std_g2, n_forecasts)
                                for _ in range(n_ensemble_members)]).T
observations_g2 = ensemble_mean_g2 + np.random.normal(0, 1.5, n_forecasts)
ranks_group2 = np.sum(ensemble_members_g2 < observations_g2[:, np.newaxis], axis=1)
df_group2 = pd.DataFrame({'rank': ranks_group2, 'group': 'Model B'})

# Combine data
df_combined = pd.concat([df_group1, df_group2]).reset_index(drop=True)

# 2. Initialize and create the plot with grouping
plot = RankHistogramPlot(figsize=(12, 7))
plot.plot(df_combined, rank_col='rank', n_members=n_ensemble_members, normalize=True, label_col='group')

# 3. Add titles and labels
plot.ax.set_title("Rank Histograms for Multiple Models")
plot.ax.set_xlabel("Rank of Observation")
plot.ax.set_ylabel("Relative Frequency")

plt.tight_layout()
plt.show()
```

### Expected Output

A single plot will display two sets of bars, one for 'Model A' and one for 'Model B', distinguished by color and an automatic legend. 'Model A' will show a flat histogram, while 'Model B' will exhibit a U-shaped histogram. This allows for a direct visual comparison of the reliability characteristics of different ensemble systems.

### Key Concepts

-   **`label_col`**: Allows plotting multiple rank histograms on the same axes, grouped by a specified categorical column.
-   **Comparative Analysis**: Facilitates easy comparison of ensemble reliability across different models or scenarios.
