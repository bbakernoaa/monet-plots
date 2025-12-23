# MONET Plots

A comprehensive plotting library for scientific visualization, spun off from the main MONET repository. MONET Plots provides a modular, extensible framework for creating high-quality scientific plots with a focus on meteorological and climate data visualization.

## Overview

MONET Plots is designed to make scientific plotting easier, more consistent, and publication-ready. It leverages the power of matplotlib, seaborn, and cartopy while providing a simplified API and consistent styling.

## Features

- **Modular Plot Classes**: Extensible plot classes for common scientific visualization needs
- **Publication-Ready Styling**: Built-in Wiley-compliant styling for professional appearance
- **Cartopy Integration**: Seamless integration with cartopy for geospatial plotting
- **Multiple Plot Types**: Spatial, time series, scatter, Taylor diagrams, KDE plots, and more
- **Flexible Configuration**: Easy customization of colors, styles, and plot parameters
- **Performance Optimized**: Designed for efficient plotting with large datasets

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Install from PyPI

```bash
pip install monet_plots
```

### Install from Source

```bash
git clone https://github.com/your-repo/monet-plots.git
cd monet-plots
pip install -e .
```

### Install Optional Dependencies

For full functionality, install additional optional dependencies:

```bash
# For geospatial plotting
pip install cartopy

# For statistical plotting
pip install seaborn pandas

# For advanced Taylor diagrams
pip install numpy matplotlib
```

## Quick Start

```python
import numpy as np
import pandas as pd
from monet_plots import SpatialPlot, TimeSeriesPlot, ScatterPlot

# Create sample data
lat = np.linspace(30, 50, 20)
lon = np.linspace(-120, -70, 30)
data = np.random.random((20, 30))

# Create a spatial plot
spatial_plot = SpatialPlot()
spatial_plot.plot(data, title="Sample Spatial Plot")
spatial_plot.save("spatial_plot.png")

# Create time series data
dates = pd.date_range('2023-01-01', periods=100, freq='D')
time_series_data = pd.DataFrame({
    'time': dates,
    'obs': np.random.normal(0, 1, 100) + np.sin(np.arange(100) * 0.1)
})

# Create a time series plot
ts_plot = TimeSeriesPlot()
ts_plot.plot(time_series_data, x='time', y='obs', title="Time Series Example")
ts_plot.save("timeseries_plot.png")
```

## Core Components

### Plot Classes

| Plot Type | Class | Description |
|-----------|-------|-------------|
| Spatial | [`SpatialPlot`](./plots/spatial) | Geospatial plots with cartopy support |
| Time Series | [`TimeSeriesPlot`](./plots/timeseries) | Time series with statistical bands |
| Scatter | [`ScatterPlot`](./plots/scatter) | Scatter plots with regression lines |
| Taylor Diagram | [`TaylorDiagramPlot`](./plots/taylor) | Model evaluation diagrams |
| KDE | [`KDEPlot`](./plots/kde) | Kernel density estimation plots |
| Wind | [`WindQuiverPlot`](./plots/wind) | Wind vector plots |
| Facet Grid | [`FacetGridPlot`](./plots/facet_grid) | Multi-panel figure layouts |

### Utility Modules

- **[`style`](./api/style)**: Publication-ready styling configuration
- **[`colorbars`](./api/colorbars)**: Custom colorbar creation utilities
- **[`taylordiagram`](./api/taylordiagram)**: Taylor diagram functionality
- **[`plot_utils`](./api/plot_utils)**: Common plotting utilities

## Basic Usage Patterns

### Creating Plots

```python
# Initialize a plot
plot = SpatialPlot(figsize=(10, 6))

# Plot data
plot.plot(data, cmap='viridis', title="My Plot")

# Save and close
plot.save("output.png")
plot.close()
```

### Customization

```python
# Custom styling
from monet_plots import wiley_style
import matplotlib.pyplot as plt

plt.style.use(wiley_style)

# Custom colorbars
from monet_plots import colorbar_index
colorbar, cmap = colorbar_index(10, 'viridis', minval=0, maxval=100)
```

## Documentation Structure

### Core Documentation

- **[Getting Started](./getting-started)**: Comprehensive installation and setup guide with detailed troubleshooting
- **[API Reference](./api)**: Complete API documentation for all modules, classes, and functions
- **[Plot Types](./plots)**: Detailed documentation for all plot types with examples and best practices

### Learning Resources

- **[Examples and Tutorials](./examples)**: Practical examples, workflows, and real-world use cases
- **[Configuration and Customization](./configuration)**: Advanced styling, theming, and customization guides
- **[Performance Optimization](./performance)**: Techniques for handling large datasets and improving speed

### Support and Troubleshooting

- **[Troubleshooting and FAQ](./troubleshooting)**: Common issues, solutions, and expert guidance
- **[Contributing Guidelines](../CONTRIBUTING.md)**: How to contribute to the project
- **[License](../LICENSE)**: Project licensing information

### Documentation Categories

| Category | Description | Level |
|----------|-------------|-------|
| **Beginner** | [Getting Started](./getting-started), [Basic Examples](./examples/getting-started) | New users |
| **Intermediate** | [API Reference](./api), [Plot Types](./plots), [Configuration](./configuration) | Regular users |
| **Advanced** | [Performance](./performance), [Advanced Examples](./examples/advanced-workflows), [Troubleshooting](./troubleshooting) | Power users |
| **Expert** | [Contributing](../CONTRIBUTING.md), [Development](../development) | Developers |

## Contributing

We welcome contributions! Please see our [contributing guidelines](../CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## Support

- Documentation: [https://monet-plots.readthedocs.io](https://monet-plots.readthedocs.io)
- Issues: [GitHub Issues](https://github.com/your-repo/monet-plots/issues)
- Discussions: [GitHub Discussions](https://github.com/your-repo/monet-plots/discussions)
