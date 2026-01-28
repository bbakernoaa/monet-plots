"""
Conditional Quantile
====================

<<<<<<<< HEAD:examples/plot_conditional_quantile.py
This example demonstrates how to create a Conditional Quantile.
========
This example demonstrates Conditional Quantile.
>>>>>>>> origin/main:examples/example_conditional_quantile.py
"""

import pandas as pd
import numpy as np
from monet_plots.plots.conditional_quantile import ConditionalQuantilePlot

# Create dummy model vs obs
obs = np.random.exponential(10, 1000)
mod = obs * 0.8 + np.random.normal(0, 5, 1000) + 2

df = pd.DataFrame({'obs': obs, 'mod': mod})

# Initialize and plot
plot = ConditionalQuantilePlot(df, obs_col='obs', mod_col='mod', bins=10)
plot.plot(show_points=True)
plot.save('conditional_quantile_example.png')
print("Conditional quantile plot saved to conditional_quantile_example.png")
