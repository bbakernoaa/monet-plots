"""
Fingerprint
===========

<<<<<<<< HEAD:examples/plot_fingerprint.py
This example demonstrates how to create a Fingerprint.
========
This example demonstrates Fingerprint.
>>>>>>>> origin/main:examples/example_fingerprint.py
"""

import pandas as pd
import numpy as np
from monet_plots.plots.fingerprint import FingerprintPlot

# Create dummy data showing a diurnal/seasonal pattern
dates = pd.date_range("2023-01-01", periods=24*365, freq="h")
hours = dates.hour
doy = dates.dayofyear
val = np.sin(2 * np.pi * hours / 24) + np.sin(2 * np.pi * doy / 365) + np.random.randn(24*365) * 0.1

df = pd.DataFrame({'time': dates, 'concentration': val})

# Initialize and plot
plot = FingerprintPlot(df, val_col='concentration', x_scale='hour', y_scale='dayofyear')
plot.plot(cmap='magma')
plot.save('fingerprint_example.png')
print("Fingerprint plot saved to fingerprint_example.png")
