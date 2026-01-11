"""
TimeSeries Plot
===============

This example demonstrates how to create a time series plot using the
:class:`monet_plots.plots.timeseries.TimeSeriesPlot` class.

The plot shows the temperature at a single location over the full time
range of the synthetic dataset.
"""
import matplotlib.pyplot as plt
import pandas as pd

from monet_plots.plots.timeseries import TimeSeriesPlot
from data import create_dataset

# 1. Create the synthetic dataset
ds = create_dataset()

# 2. Select data for a single point
# Extract the temperature at the lowest level for a specific lat/lon.
# The TimeSeriesPlot can now handle this xarray.DataArray directly.
da = ds["temperature"].isel(level=0, latitude=5, longitude=5)

# 3. Create a TimeSeriesPlot instance and plot the data
fig, ax = plt.subplots(figsize=(10, 6))
# Pass the DataArray directly to the plot class
plot = TimeSeriesPlot(ax=ax, data=da, y="temperature")
plot.plot(
    label="Surface Temperature",
    color="blue",
    marker="o",
)

# 4. Customize the plot
ax.set_title("Surface Temperature Time Series at a Single Point")
ax.set_ylabel(f"Temperature ({da.attrs['units']})")
ax.set_xlabel("Time")
ax.grid(True, linestyle="--", alpha=0.6)

# Improve date formatting on the x-axis
fig.autofmt_xdate()

# Let mkdocs-gallery display the plot
plt.show()
