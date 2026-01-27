"""
Polar
=====

This example demonstrates how to create a Polar.
"""

import pandas as pd
import numpy as np
from monet_plots.plots.polar import BivariatePolarPlot

# Create dummy wind dependent data
n = 1000
ws = np.random.gamma(2, 2, n)
wd = np.random.uniform(0, 360, n)
# Concentration higher when wind is from the East (90 deg) and low speed
conc = 10 * np.exp(-((wd - 90)**2) / 1000) * np.exp(-ws / 5) + np.random.rand(n)

df = pd.DataFrame({'ws': ws, 'wd': wd, 'conc': conc})

# Initialize and plot
plot = BivariatePolarPlot(df, ws_col='ws', wd_col='wd', val_col='conc')
plot.plot(n_bins_ws=15, n_bins_wd=36)
plot.save('polar_example.png')
print("Bivariate polar plot saved to polar_example.png")
