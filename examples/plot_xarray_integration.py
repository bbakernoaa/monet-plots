"""
Xarray Integration
==================

This example demonstrates how to create a Xarray Integration.
"""

import xarray as xr
import pandas as pd
import numpy as np
from monet_plots.plots.timeseries import TimeSeriesPlot

# Create sample xarray data
dates = pd.date_range('2023-01-01', periods=100, freq='h')
temperature = 15 + 5 * np.sin(np.arange(100) * 2 * np.pi / 24) + np.random.normal(0, 1, 100)

da = xr.DataArray(
    temperature,
    dims=['time'],
    coords={'time': dates},
    name='temperature',
    attrs={'units': '°C', 'long_name': 'Air Temperature'}
)

# Create and plot
plot = TimeSeriesPlot(da, x='time', y='temperature',
                     title="Temperature Time Series",
                     ylabel="Temperature (°C)")
ax = plot.plot()
plot.save("temperature_timeseries.png")
plot.close()
