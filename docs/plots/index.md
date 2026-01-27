# Plot Types Documentation

## Overview

This document serves as the central index for all available **Verification Plots** within the MONET Plots system. These plots are specifically designed to evaluate model forecast quality against observations across various statistical and spatial domains. They provide critical insights into bias, reliability, skill, and resolution, adhering to best practices in meteorological and statistical verification.

For detailed instructions on each plot's usage, customization, and underlying methodology, refer to the individual documentation pages listed below.

## Available Verification Plots

The following verification plots are available. Each link directs to a page detailing its usage, configuration, and interpretation.

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

For more details on styling, please see the [Configuration Guide](../configuration/index.md).
