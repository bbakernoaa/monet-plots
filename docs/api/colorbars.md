# Colorbars Module

The `monet_plots.colorbars` module provides utilities for creating and customizing colorbars in scientific plots. These utilities are designed to work seamlessly with MONET Plots and matplotlib.

## Overview

Colorbars are essential for interpreting spatial and statistical plots. This module provides advanced colorbar creation functionality including indexed colorbars, custom tick labels, and colormap manipulation.

## Functions

### `colorbar_index(ncolors, cmap, minval=None, maxval=None, dtype="int", basemap=None)`

Create a colorbar with discrete colors and custom tick labels.

```python
from monet_plots import colorbar_index

# Create indexed colorbar
cbar, cmap = colorbar_index(
    ncolors=10,
    cmap='viridis',
    minval=0,
    maxval=100
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ncolors` | `int` | Required | Number of discrete colors to use |
| `cmap` | `str` or `Colormap` | Required | Colormap name or object |
| `minval` | `float` | `None` | Minimum value for tick labels |
| `maxval` | `float` | `None` | Maximum value for tick labels |
| `dtype` | `str` or `type` | `"int"` | Data type for tick labels |
| `basemap` | `Basemap` | `None` | Basemap instance (optional) |

**Returns:**
- `colorbar`: matplotlib.colorbar.Colorbar instance
- `discretized_cmap`: Discretized colormap

**Example:**
```python
import numpy as np
from monet_plots import SpatialPlot, colorbar_index

# Create spatial plot
plot = SpatialPlot(figsize=(10, 8))

# Generate sample data
data = np.random.random((50, 100)) * 100

# Plot with indexed colorbar
im = plot.ax.imshow(data, cmap='viridis')

# Create indexed colorbar
cbar, cmap = colorbar_index(
    ncolors=10,
    cmap='viridis',
    minval=0,
    maxval=100,
    dtype=int
)

# Use the discretized colormap for the plot
im.set_cmap(cmap)
plot.save('indexed_colorbar.png')
```

### `cmap_discretize(cmap, N)`

Return a discrete colormap from a continuous colormap.

```python
from monet_plots import cmap_discretize

# Discretize continuous colormap
discrete_cmap = cmap_discretize('viridis', 5)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cmap` | `str` or `Colormap` | Required | Colormap name or object to discretize |
| `N` | `int` | Required | Number of discrete colors |

**Returns:**
- `matplotlib.colors.LinearSegmentedColormap`: Discretized colormap

**Example:**
```python
import numpy as np
import matplotlib.pyplot as plt
from monet_plots import cmap_discretize, SpatialPlot

# Create discretized colormap
discrete_cmap = cmap_discretize('plasma', 7)

# Use in plot
plot = SpatialPlot()
data = np.random.random((30, 50))
plot.plot(data, cmap=discrete_cmap, title="Discrete Colormap")
plot.save('discrete_colormap.png')
```

### `colorbar_from_cmap(cmap, vmin=None, vmax=None, **kwargs)`

Create a colorbar directly from a colormap.

```python
from monet_plots import colorbar_from_cmap

# Create colorbar from colormap
cbar = colorbar_from_cmap('viridis', vmin=0, vmax=1)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cmap` | `str` or `Colormap` | Required | Colormap name or object |
| `vmin` | `float` | `None` | Minimum value for colorbar |
| `vmax` | `float` | `None` | Maximum value for colorbar |
| `**kwargs` | `dict` | `{}` | Additional colorbar parameters |

**Returns:**
- `matplotlib.colorbar.Colorbar`: Colorbar object

**Example:**
```python
from monet_plots import colorbar_from_cmap, SpatialPlot

plot = SpatialPlot()
data = np.random.random((40, 60))

# Create plot with custom colorbar
im = plot.ax.imshow(data, cmap='RdBu_r', vmin=-1, vmax=1)

# Add colorbar
cbar = colorbar_from_cmap(
    'RdBu_r',
    vmin=-1,
    vmax=1,
    label='Temperature Anomaly'
)

plot.save('custom_colorbar.png')
```

### `add_colorbar(ax, im, label='', **kwargs)`

Add a colorbar to an existing axes object.

```python
from monet_plots import add_colorbar

# Add colorbar to existing plot
cbar = add_colorbar(plot.ax, im, label='Value')
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ax` | `matplotlib.axes.Axes` | Required | Axes object to add colorbar to |
| `im` | `matplotlib.image.AxesImage` | Required | Image object to create colorbar from |
| `label` | `str` | `''` | Colorbar label |
| `**kwargs` | `dict` | `{}` | Additional colorbar parameters |

**Returns:**
- `matplotlib.colorbar.Colorbar`: Colorbar object

**Example:**
```python
import numpy as np
from monet_plots import SpatialPlot, add_colorbar

plot = SpatialPlot()
data = np.random.random((35, 55))

# Create image
im = plot.ax.imshow(data, cmap='coolwarm')

# Add colorbar with custom formatting
cbar = add_colorbar(
    plot.ax,
    im,
    label='Concentration (ppb)',
    orientation='horizontal',
    shrink=0.8
)

plot.save('horizontal_colorbar.png')
```

### `create_diverging_cmap(n_colors=256, center=0, **kwargs)`

Create a diverging colormap centered at a specific value.

```python
from monet_plots import create_diverging_cmap

# Create custom diverging colormap
div_cmap = create_diverging_cmap(n_colors=256, center=0.5)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_colors` | `int` | `256` | Number of colors in colormap |
| `center` | `float` | `0` | Center value for divergence |
| `**kwargs` | `dict` | `{}` | Additional colormap parameters |

**Returns:**
- `matplotlib.colors.LinearSegmentedColormap`: Diverging colormap

**Example:**
```python
import numpy as np
from monet_plots import SpatialPlot, create_diverging_cmap

plot = SpatialPlot()
data = np.random.normal(0, 1, (40, 60))

# Create diverging colormap
div_cmap = create_diverging_cmap(
    n_colors=128,
    center=0,
    name='custom_diverging'
)

# Plot with diverging colormap
plot.plot(data, cmap=div_cmap, title="Diverging Colormap")
plot.save('diverging_colormap.png')
```

## Advanced Colorbar Techniques

### Custom Tick Labels

```python
from monet_plots import colorbar_index
import numpy as np

# Create colorbar with custom labels
cbar, cmap = colorbar_index(
    ncolors=5,
    cmap='YlOrRd',
    minval=0,
    maxval=50,
    dtype=float
)

# Customize tick labels manually
cbar.set_ticks([0, 10, 20, 30, 40, 50])
cbar.set_ticklabels(['Low', 'Low-Med', 'Medium', 'Med-High', 'High'])
```

### Colorbar with Scientific Notation

```python
from monet_plots import colorbar_from_cmap
import matplotlib.ticker as ticker

# Create colorbar with scientific notation
cbar = colorbar_from_cmap('viridis', vmin=1e-6, vmax=1e2)

# Format ticks with scientific notation
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-3, 3))
cbar.ax.yaxis.set_major_formatter(formatter)
```

### Colorbar with Multiple Levels

```python
from monet_plots import colorbar_index, cmap_discretize

# Create multi-level colorbar
levels = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
n_colors = len(levels) - 1

cbar, cmap = colorbar_index(
    ncolors=n_colors,
    cmap='RdYlBu_r',
    minval=levels[0],
    maxval=levels[-1],
    dtype=int
)

# Set custom tick positions
cbar.set_ticks(levels)
```

## Common Use Cases

### Spatial Plots

```python
import numpy as np
from monet_plots import SpatialPlot, colorbar_index

# Create spatial plot with indexed colorbar
plot = SpatialPlot(figsize=(10, 8))
data = np.random.random((50, 100)) * 150

# Plot data
im = plot.ax.imshow(data, cmap='viridis')

# Create indexed colorbar
cbar, cmap = colorbar_index(
    ncolors=15,
    cmap='viridis',
    minval=0,
    maxval=150,
    dtype=int
)

# Update plot colormap
im.set_cmap(cmap)

# Add labels and title
plot.title("Spatial Distribution", fontsize=14)
plot.xlabel("Longitude")
plot.ylabel("Latitude")

plot.save("spatial_with_colorbar.png")
```

### Statistical Plots

```python
import numpy as np
from monet_plots import colorbar_from_cmap, ScatterPlot

# Create scatter plot with custom colorbar
plot = ScatterPlot()
x = np.random.normal(0, 1, 1000)
y = np.random.normal(0, 1, 1000)
colors = np.random.uniform(0, 100, 1000)

scatter = plot.ax.scatter(x, y, c=colors, cmap='plasma', alpha=0.6)

# Add colorbar
cbar = colorbar_from_cmap(
    'plasma',
    vmin=0,
    vmax=100,
    label='Confidence Level'
)

plot.title("Scatter Plot with Color Mapping")
plot.xlabel("X Variable")
plot.ylabel("Y Variable")

plot.save("scatter_with_colorbar.png")
```

### Time Series Plots

```python
import numpy as np
import pandas as pd
from monet_plots import TimeSeriesPlot, colorbar_index

# Create time series data
dates = pd.date_range('2023-01-01', periods=365, freq='D')
data = pd.DataFrame({
    'date': dates,
    'temperature': 15 + 10 * np.sin(np.arange(365) * 0.1) + np.random.normal(0, 2, 365),
    'confidence': np.random.uniform(0.8, 1.0, 365)
})

# Create time series plot with colored confidence bands
plot = TimeSeriesPlot(figsize=(12, 6))

# Plot temperature with confidence-based coloring
for i in range(len(data) - 1):
    color_value = data['confidence'].iloc[i]
    plot.ax.plot(
        data['date'].iloc[i:i+2],
        data['temperature'].iloc[i:i+2],
        color=plt.cm.RdYlGn(color_value),
        linewidth=2
    )

# Add colorbar for confidence
cbar, cmap = colorbar_index(
    ncolors=10,
    cmap='RdYlGn',
    minval=0.8,
    maxval=1.0,
    dtype=float
)

plot.title("Temperature with Confidence Bands")
plot.xlabel("Date")
plot.ylabel("Temperature (°C)")

plot.save("timeseries_with_colorbar.png")
```

## Performance Considerations

- **Reuse colorbars**: Create colorbars once and reuse when possible
- **Limit discrete levels**: Use reasonable numbers of discrete colors (5-20)
- **Cache colormaps**: Store discretized colormaps for repeated use

## Error Handling

The module includes automatic error handling for common issues:

- **Invalid colormaps**: Provides helpful error messages for unknown colormap names
- **Invalid parameters**: Validates input ranges and data types
- **Missing dependencies**: Graceful handling of missing matplotlib components

## Troubleshooting

### Colorbar Not Visible

```python
# Ensure colorbar is properly attached
cbar = plt.colorbar(im, ax=plot.ax)  # Explicitly specify axes
```

### Incorrect Tick Labels

```python
# Manually set tick labels
cbar.set_ticks([0, 25, 50, 75, 100])
cbar.set_ticklabels(['Min', 'Q1', 'Median', 'Q3', 'Max'])
```

### Colorbar Alignment Issues

```python
# Adjust colorbar position
cbar.ax.set_position([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
```

---

**Related Resources**:

- [Style Configuration](./style) - Plot styling and themes
- [Plot Types API](../plots) - Specific plot implementations
- [Examples](../examples) - Practical usage examples
