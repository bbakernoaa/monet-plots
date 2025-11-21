# Verification Plots Documentation

## Overview

This document serves as the central index for all available **Verification Plots** within the MONET Plots system. These plots are specifically designed to evaluate model forecast quality against observations across various statistical and spatial domains. They provide critical insights into bias, reliability, skill, and resolution, adhering to best practices in meteorological and statistical verification.

For detailed instructions on each plot's usage, customization, and underlying methodology, refer to the individual documentation pages listed below.

## Available Verification Plots

The following verification plots are available. Each link directs to a page detailing its usage, configuration, and interpretation.

| Plot Name | Description | Documentation Link |
| :--- | :--- | :--- |
| [`PerformanceDiagramPlot`](./performance_diagram) | Analyzes forecast performance across various thresholds, often combining skill and bias metrics. | [`Performance Diagram`](./performance_diagram) |
| [`ROCCurvePlot`](./roc_curve) | Generates a Receiver Operating Characteristic (ROC) curve to visualize the trade-off between hit rate and false alarm rate. | [`ROC Curve`](./roc_curve) |
| [`ReliabilityDiagramPlot`](./reliability_diagram) | Assesses calibration by comparing forecast probability against observed relative frequency. | [`Reliability Diagram`](./reliability_diagram) |
| [`RankHistogramPlot`](./rank_histogram) | Evaluates whether forecast ensembles are properly spread by analyzing the rank of observations within the ensemble members. | [`Rank Histogram`](./rank_histogram) |
| [`BrierScoreDecompositionPlot`](./brier_decomposition) | Decomposes the Brier Score into components representing uncertainty, resolution, and reliability. | [`Brier Score Decomposition`](./brier_decomposition) |
| [`ScorecardPlot`](./scorecard) | Provides a tabular or graphical summary of key verification statistics for quick assessment. | [`Scorecard`](./scorecard) |
| [`RelativeEconomicValuePlot`](./rev) | Evaluates the economic value of forecasts by comparing them against reference forecasts. | [`Relative Economic Value (REV)`](./rev) |
| [`ConditionalBiasPlot`](./conditional_bias) | Shows bias conditioned on forecast value, helping to identify systematic over/under-forecasting for certain outcomes. | [`Conditional Bias`](./conditional_bias) |

## Usage and Style Guidelines

All verification plots follow the same core structure:

1.  **Initialization**: Instantiate the plot class (e.g., `PerformanceDiagramPlot(...)`).
2.  **Plotting**: Call the main rendering method, passing in the required observation and forecast data.
3.  **Customization**: Use methods like `.title()`, `.xlabel()`, or global configuration to style the plot.
4.  **Output**: Save the figure using `.save()` and always remember to close it with `.close()` to manage memory effectively, as documented in the [API Reference](./api/base).

For more details on styling, please see the [Configuration Guide](./configuration).