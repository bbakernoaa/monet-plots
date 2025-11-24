# Conditional Bias Plot

The `ConditionalBiasPlot` visualizes the bias (Forecast - Observation) as a function of the Observed Value. This is also known as a Conditional Quantile Plot or Bias-Variance decomposition plot in some contexts.

This plot helps to identify if the model's bias is dependent on the magnitude of the observed values.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.monet_plots.plots.conditional_bias import ConditionalBiasPlot

# Generate sample data
np.random.seed(42)
n_samples = 200
observations = np.random.rand(n_samples) * 100
forecasts = observations + np.random.randn(n_samples) * 10 - 5 # Forecasts with some bias

# Create a DataFrame
data = pd.DataFrame({
    'observations': observations,
    'forecasts': forecasts
})

# Create the plot
fig, ax = plt.subplots(figsize=(8, 6))
plotter = ConditionalBiasPlot(fig=fig, ax=ax)
plotter.plot(data, obs_col='observations', fcst_col='forecasts', n_bins=10)

# Display the plot (in a real script, you might save it)
plt.tight_layout()
plt.show()
```

![Conditional Bias Plot](conditional_bias_plot.png)
