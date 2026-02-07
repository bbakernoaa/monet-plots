"""
Time Series Statistics Plot
===========================

**What it's for:**
Creating time series plots of performance metrics (e.g., Bias, RMSE, Correlation) resampled to a specific frequency.

**When to use:**
Use this to evaluate model performance over time, detecting seasonal patterns or trends in model error.

**How to read:**
The x-axis represents time (resampled), and the y-axis represents the calculated statistic between observations and model predictions.
"""

import xarray as xr
import pandas as pd
import numpy as np
import dask.array as da
import matplotlib.pyplot as plt
from monet_plots.plots.timeseries import TimeSeriesStatsPlot

# 1. Create sample Xarray data (Lazy by Default using Dask)
ntime = 1000
dates = pd.date_range('2023-01-01', periods=ntime, freq='h')

# Observation data
obs_data = 15 + 5 * np.sin(np.arange(ntime) * 2 * np.pi / 24) + np.random.normal(0, 1, ntime)
obs = xr.DataArray(
    da.from_array(obs_data, chunks=200),
    dims=['time'],
    coords={'time': dates},
    name='Obs'
)

# Model data (with some bias and noise)
mod_data = obs_data + 2.0 + np.random.normal(0, 2, ntime)
mod = xr.DataArray(
    da.from_array(mod_data, chunks=200),
    dims=['time'],
    coords={'time': dates},
    name='Model'
)

# Merge into a dataset
ds = xr.Dataset({'Obs': obs, 'Model': mod})

# 2. Initialize and Plot Bias (Daily Resampling)
plot = TimeSeriesStatsPlot(ds, col1='Obs', col2='Model', figsize=(10, 5))
ax = plot.plot(stat='bias', freq='D', color='red', label='Daily Bias')

# 3. Add titles and labels
plot.ax.set_title("Daily Mean Bias (Model - Obs)")
plot.ax.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# 4. Show RMSE with Weekly Resampling
plot_rmse = TimeSeriesStatsPlot(ds, col1='Obs', col2='Model', figsize=(10, 5))
plot_rmse.plot(stat='rmse', freq='W', color='blue', label='Weekly RMSE')
plot_rmse.ax.set_title("Weekly Root Mean Square Error")
plt.tight_layout()
plt.show()
