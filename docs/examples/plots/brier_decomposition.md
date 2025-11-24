# Brier Score Decomposition Plot

The Brier Score Decomposition Plot visualizes the components of the Brier Score: Reliability, Resolution, and Uncertainty. The Brier Score (BS) can be decomposed as BS = Reliability - Resolution + Uncertainty.

This plot helps in understanding the contributions of different factors to the overall forecast accuracy.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.monet_plots.plots.brier_decomposition import BrierScoreDecompositionPlot

# Generate sample data
np.random.seed(42)
n_samples = 1000
forecast_probabilities = np.random.rand(n_samples)
observations = np.random.randint(0, 2, n_samples)

# Create a DataFrame
data = pd.DataFrame({
    'forecasts': forecast_probabilities,
    'observations': observations
})

# Create the plot
fig, ax = plt.subplots(figsize=(8, 6))
plotter = BrierScoreDecompositionPlot(fig=fig, ax=ax)
plotter.plot(data, forecasts_col='forecasts', observations_col='observations', n_bins=5)

# Display the plot (in a real script, you might save it)
plt.tight_layout()
plt.show()
```

![Brier Score Decomposition Plot](brier_decomposition.png)
