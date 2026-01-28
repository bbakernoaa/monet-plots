"""
Diurnal Error
=============

<<<<<<<< HEAD:examples/plot_diurnal_error.py
This example demonstrates how to create a Diurnal Error.
========
This example demonstrates Diurnal Error.
>>>>>>>> origin/main:examples/example_diurnal_error.py
"""

import pandas as pd
import numpy as np
from monet_plots.plots.diurnal_error import DiurnalErrorPlot

# Create dummy time series data
dates = pd.date_range("2023-01-01", periods=24*30, freq="h")
df = pd.DataFrame({
    'time': dates,
    'obs': np.random.rand(24*30) * 10 + 5,
    'mod': np.random.rand(24*30) * 10 + 6  # Slight bias
})

# Initialize and plot
plot = DiurnalErrorPlot(df, obs_col='obs', mod_col='mod', second_dim='dayofweek')
plot.plot()
plot.save('diurnal_error_example.png')
print("Diurnal error plot saved to diurnal_error_example.png")
