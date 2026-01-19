# Plot Types Documentation

## Overview

This document serves as the central index for all available plot types within the MONET Plots system. These plots are categorized into groups based on their primary use case.

## Verification Plots

These plots are specifically designed to evaluate model forecast quality against observations.

| Plot Name | Description | Documentation Link |
| :--- | :--- | :--- |
| [`PerformanceDiagramPlot`](./performance_diagram.md) | Analyzes forecast performance across POD, Success Ratio, and CSI. | [`Performance Diagram`](./performance_diagram.md) |
| [`ROCCurvePlot`](./roc_curve.md) | Visualizes the trade-off between hit rate and false alarm rate. | [`ROC Curve`](./roc_curve.md) |
| [`ReliabilityDiagramPlot`](./reliability_diagram.md) | Assesses calibration of probabilistic forecasts. | [`Reliability Diagram`](./reliability_diagram.md) |
| [`RankHistogramPlot`](./rank_histogram.md) | Evaluates ensemble spread and reliability. | [`Rank Histogram`](./rank_histogram.md) |
| [`BrierScoreDecompositionPlot`](./brier_decomposition.md) | Decomposes Brier Score into components. | [`Brier Score Decomposition`](./brier_decomposition.md) |
| [`ScorecardPlot`](./scorecard.md) | Summary of key verification statistics. | [`Scorecard`](./scorecard.md) |
| [`RelativeEconomicValuePlot`](./rev.md) | Evaluates the economic value of forecasts. | [`Relative Economic Value (REV)`](./rev.md) |
| [`ConditionalBiasPlot`](./conditional_bias.md) | Shows bias conditioned on forecast value. | [`Conditional Bias`](./conditional_bias.md) |
| [`SoccerPlot`](./soccer.md) | Plots model bias against error with target zones. | [`Soccer Plot`](./soccer.md) |
| [`ConditionalQuantilePlot`](./conditional_quantile.md) | Modeled value quantiles conditioned on observations. | [`Conditional Quantile Plot`](./conditional_quantile.md) |

## Spatial Plots

Specialized plots for visualizing geospatial data on maps.

| Plot Name | Description | Documentation Link |
| :--- | :--- | :--- |
| [`SpatialPlot`](./spatial.md) | Base class for geospatial plots with cartopy support. | [`Spatial Plot`](./spatial.md) |
| [`SpatialContourPlot`](./spatial_contour.md) | 2D contour plots on a geographical map. | [`Spatial Contour`](./spatial_contour.md) |
| [`SpatialImshowPlot`](./spatial_imshow.md) | Gridded spatial data displayed as an image on a map. | [`Spatial Imshow`](./spatial_imshow.md) |
| [`SpatialBiasScatterPlot`](./spatial_bias_scatter.md) | Geographical distribution of bias with points. | [`Spatial Bias Scatter`](./spatial_bias_scatter.md) |
| [`SpScatterBiasPlot`](./sp_scatter_bias.md) | Alternative spatial bias visualization. | [`SP Scatter Bias`](./sp_scatter_bias.md) |

## Basic & Statistical Plots

Fundamental plot types for general data analysis.

| Plot Name | Description | Documentation Link |
| :--- | :--- | :--- |
| [`TimeSeriesPlot`](./timeseries.md) | Plot data over time with statistical bands. | [`Time Series Plot`](./timeseries.md) |
| [`ScatterPlot`](./scatter.md) | Relationship between two variables with regression. | [`Scatter Plot`](./scatter.md) |
| [`KDEPlot`](./kde.md) | Kernel density estimation for probability distributions. | [`KDE Plot`](./kde.md) |
| [`CategoricalPlot`](./categorical.md) | Bar or violin plots for categorical data. | [`Categorical Plot`](./categorical.md) |
| [`RidgelinePlot`](./ridgeline.md) | Distribution curves for several groups (joyplot). | [`Ridgeline Plot`](./ridgeline.md) |
| [`TaylorDiagramPlot`](./taylor_diagram.md) | Summarizes standard deviation, correlation, and RMSE. | [`Taylor Diagram`](./taylor_diagram.md) |
| [`FacetGridPlot`](./facet_grid.md) | Multi-panel figure layouts using Seaborn. | [`Facet Grid`](./facet_grid.md) |

## Meteorological & Specialized Plots

| Plot Name | Description | Documentation Link |
| :--- | :--- | :--- |
| [`WindQuiverPlot`](./wind.md) | Wind vector arrows indicating direction and magnitude. | [`Wind Quiver`](./wind.md) |
| [`WindBarbsPlot`](./wind_barbs.md) | Conventional meteorological wind barbs. | [`Wind Barbs`](./wind_barbs.md) |
| [`CurtainPlot`](./curtain.md) | Vertical cross-section (altitude vs time/distance). | [`Vertical Curtain Plot`](./curtain.md) |
| [`DiurnalErrorPlot`](./diurnal_error.md) | Heat map of model error by hour of day. | [`Diurnal Error Plot`](./diurnal_error.md) |
| [`FingerprintPlot`](./fingerprint.md) | Temporal patterns across two different scales. | [`Fingerprint Plot`](./fingerprint.md) |
| [`BivariatePolarPlot`](./polar.md) | Dependence on wind speed and direction. | [`Bivariate Polar Plot`](./polar.md) |
| [`ProfilePlot`](./profile.md) | Vertical atmospheric profiles. | [`Profile Plot`](./profile.md) |

## Usage and Style Guidelines

All plot classes follow the same core structure:

1.  **Initialization**: Instantiate the plot class (e.g., `TimeSeriesPlot(df, ...)`).
2.  **Plotting**: Call the main `.plot()` method.
3.  **Customization**: Use methods like `.ax.set_title()` or global configuration.
4.  **Output**: Save the figure using `.save()` and close with `.close()`.

For more details on styling, please see the [Configuration Guide](../../configuration/index.md).
