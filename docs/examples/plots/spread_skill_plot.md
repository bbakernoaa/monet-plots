# Spread-Skill Plot

The `SpreadSkillPlot` evaluates ensemble forecast reliability by comparing the standard deviation of the ensemble spread to the root mean squared error (RMSE) of the ensemble mean. A reliable ensemble should have a spread that is proportional to the forecast error.

This plot helps in assessing the consistency between the ensemble's predicted uncertainty and its actual error.

```python
import numpy as np
import matplotlib.pyplot as plt
from src.monet_plots.plots.ensemble import SpreadSkillPlot

# Generate sample data
np.random.seed(42)
spread_values = np.random.rand(50) * 10 + 5
skill_values = spread_values + np.random.randn(50) * 2

# Create the plot
fig, ax = plt.subplots(figsize=(7, 7))
plotter = SpreadSkillPlot(spread=spread_values, skill=skill_values, fig=fig, ax=ax)
plotter.plot(alpha=0.7, color='blue', s=50)

# Display the plot (in a real script, you might save it)
plt.tight_layout()
plt.show()
```

![Spread-Skill Plot](spread_skill_plot.png)
