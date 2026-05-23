# Plot Types Documentation

## Overview

This document serves as the central index for all available **Verification and Spatial Plots** within the MONET Plots system. These plots are specifically designed to evaluate model forecast quality against observations and visualize geospatial data following best practices in Earth science data engineering.

For detailed instructions on each plot's usage, customization, and underlying methodology, refer to the individual documentation pages or explore the [Examples Gallery](../gallery/index.md).

## Spatial Plots

These plots are used for visualizing data on geographic maps.

| Plot Name | Description | Documentation Link |
| :--- | :--- | :--- |
| `SpatialPlot` | Base class for all spatial plots, providing cartopy integration. | [`Spatial`](./spatial.md) |
| `SpatialContourPlot` | Filled or line contours of 2D gridded data. | [`Spatial Contour`](./spatial_contour.md) |
| `SpatialImshowPlot` | High-performance pixel-based visualization of 2D grids. | [`Spatial Imshow`](./spatial_imshow.md) |
| `SpatialBiasScatterPlot` | Point observations colored by model bias on a map. | [`Spatial Bias Scatter`](./spatial_bias_scatter.md) |
| `SpatialTrack` | Path-based data (e.g., flight tracks) on a map. | [`Spatial Track`](./spatial.md) |
| `SpatialOverlayPlot` | Point observations overlayed on a gridded model field. | [`Spatial Overlay`](./spatial.md) |
| `UpperAir` | Specialized plots for upper-air meteorological data. | [`Upper Air`](./upper_air.md) |

## Verification & Statistical Plots

These plots are used to evaluate model performance against observations.

| Plot Name | Description | Documentation Link |
| :--- | :--- | :--- |
| `TimeSeriesPlot` | Temporal evolution of variables. | [`Time Series`](./timeseries.md) |
| `TimeSeriesStatsPlot` | Time series with statistical context (e.g., ensemble spread). | [`Time Series`](./timeseries.md) |
| `ScatterPlot` | One-to-one comparison of model and observations. | [`Scatter`](./scatter.md) |
| `KDEPlot` | Univariate and bivariate Kernel Density Estimation. | [`KDE`](./kde.md) |
| `DiurnalErrorPlot` | Heat map of model error by hour of day. | [`Diurnal Error`](./diurnal_error.md) |
| `PerformanceDiagramPlot` | Summary of categorical forecast performance. | [`Performance Diagram`](./performance_diagram.md) |
| `TaylorDiagramPlot` | Standardized visualization of correlation, RMSE, and SD. | [`Taylor Diagram`](./taylor_diagram.md) |
| `SoccerPlot` | Specialized plot for bias and error evaluation. | [`Soccer Plot`](./soccer.md) |
| `ScorecardPlot` | Grid-based summary of multiple metrics and models. | [`Scorecard`](./scorecard.md) |
| `RankHistogram` | Evaluation of ensemble reliability. | [`Rank Histogram`](./rank_histogram.md) |
| `ReliabilityDiagram` | Calibration of probabilistic forecasts. | [`Reliability`](./reliability_diagram.md) |
| `ROCCurvePlot` | Discrimination ability of probabilistic forecasts. | [`ROC Curve`](./roc_curve.md) |
| `ConditionalBiasPlot` | Model bias conditioned on other variables. | [`Conditional Bias`](./conditional_bias.md) |
| `ConditionalQuantilePlot` | Evaluation of the entire distribution's bias. | [`Conditional Quantile`](./conditional_quantile.md) |
| `BrierScoreDecompositionPlot` | Decomposition of the Brier Score for probability forecasts. | [`Brier Decomposition`](./brier_decomposition.md) |
| `RelativeEconomicValuePlot` | Economic value of forecasts for decision-making. | [`REV`](./rev.md) |

## Profile & Atmospheric Plots

These plots are used for vertical and regional atmospheric analysis.

| Plot Name | Description | Documentation Link |
| :--- | :--- | :--- |
| `ProfilePlot` | Basic vertical atmospheric profile. | [`Profile`](./profile.md) |
| `VerticalProfilePlot` | Binned vertical statistics comparing models and obs. | [`Profile`](./profile.md) |
| `CurtainPlot` | Vertical cross-section (altitude vs time/distance). | [`Curtain`](./curtain.md) |
| `VerticalSlice` | Cross-sectional contour plot of 2D slices. | [`Profile`](./profile.md) |
| `StickPlot` | Vertical wind vectors. | [`Profile`](./profile.md) |
| `VerticalBoxPlot` | Statistical distributions by altitude bin. | [`Profile`](./profile.md) |
| `Meteogram` | Stacked time series of multiple variables at one site. | [`Meteogram`](./meteogram.md) |
| `Windrose` | Frequency and intensity of wind direction and speed. | [`Wind`](./wind.md) |
| `WindBarbsPlot` | Meteorological wind barbs on a map. | [`Wind Barbs`](./wind_barbs.md) |
| `WindQuiverPlot` | Wind vectors (arrows) on a map. | [`Wind Quiver`](./wind.md) |
| `BivariatePolarPlot` | Variable dependence on wind speed and direction. | [`Polar`](./polar.md) |

## Advanced Layouts

| Plot Name | Description | Documentation Link |
| :--- | :--- | :--- |
| `FacetGridPlot` | Multi-panel grid for categorical data subsets. | [`Facet Grid`](./facet_grid.md) |
| `SpatialFacetGridPlot` | Multi-panel grid of geographic maps. | [`Facet Grid`](./facet_grid.md) |
| `TrajectoryPlot` | Combined map and time series for moving platforms. | [`Spatial`](./spatial.md) |
| `RidgelinePlot` | Overlapping density plots for visualizing distributions over time or space. | [`Ridgeline`](./ridgeline.md) |

## Usage and Style Guidelines

All plot classes follow the same core structure:

1.  **Initialization**: Instantiate the plot class (e.g., `TimeSeriesPlot(df, ...)`).
2.  **Plotting**: Call the main `.plot()` method.
3.  **Customization**: Use methods like `.ax.set_title()` or global configuration.
4.  **Output**: Save the figure using `.save()` and close with `.close()`.

For more details on styling, including the **Pivotal Weather** and **WeatherMesh** styles, please see the [Configuration Guide](../configuration/index.md).
