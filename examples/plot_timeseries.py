"""
Time Series Plot
================

This example demonstrates how to create a time series plot.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from monet_plots.plots.timeseries import TimeSeriesPlot

dates = pd.date_range('2023-01-01', periods=100, freq='D')
values = np.cumsum(np.random.normal(0, 1, 100)) + 50
df = pd.DataFrame({'time': dates, 'values': values})
plot = TimeSeriesPlot(df=df, figsize=(12, 6))
plot.plot(x='time', y='values', title="Daily Time Series", ylabel="Temperature (°C)")
plt.show()
