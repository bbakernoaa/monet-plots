# Time Series Plots

The `timeseries` module provides classes for visualizing data over time, including raw time series and statistical evaluations.

## TimeSeriesPlot

The `TimeSeriesPlot` class creates a standard time series plot with shaded error bounds (±1 standard deviation).

::: monet_plots.plots.timeseries.TimeSeriesPlot

## TimeSeriesStatsPlot

The `TimeSeriesStatsPlot` class allows for plotting performance metrics (e.g., Bias, RMSE, Correlation) resampled to a specific frequency. It supports both Pandas and Xarray/Dask objects, ensuring efficient processing of large datasets.

::: monet_plots.plots.timeseries.TimeSeriesStatsPlot
