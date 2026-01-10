# Plot Utils Module

The `monet_plots.plot_utils` module provides utility functions for common plotting tasks that are used across multiple plot types. These utilities help maintain consistency and provide convenience functions for complex operations.

## Overview

This module contains helper functions for formatting axes, managing figures, and performing common plotting operations that are shared across different plot types.

## Functions

### `add_colorbar(ax, im, label='', orientation='vertical', fraction=0.15, pad=0.04, **kwargs)`

Add a colorbar to an existing axes.

```python
from monet_plots.plot_utils import add_colorbar

# Add colorbar to existing plot
cbar = add_colorbar(plot.ax, im, label='Temperature', orientation='horizontal')
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ax` | `matplotlib.axes.Axes` | Required | Axes object to add colorbar to |
| `im` | `matplotlib.image.AxesImage` | Required | Image object |
| `label` | `str` | `''` | Colorbar label |
| `orientation` | `str` | `'vertical'` | Colorbar orientation |
| `fraction` | `float` | `0.15` | Fraction of figure size |
| `pad` | `float` | `0.04` | Padding between axes and colorbar |
| `**kwargs` | `dict` | `{}` | Additional colorbar parameters |

**Returns:**
- `matplotlib.colorbar.Colorbar`: Colorbar object

**Example:**
```python
import numpy as np
from monet_plots import SpatialPlot
from monet_plots.plot_utils import add_colorbar

plot = SpatialPlot(figsize=(10, 8))
data = np.random.random((50, 100))

# Create image
im = plot.ax.imshow(data, cmap='viridis')

# Add colorbar with custom formatting
cbar = add_colorbar(
    plot.ax,
    im,
    label='Concentration (ppb)',
    orientation='vertical',
    fraction=0.12,
    pad=0.02,
    shrink=0.8
)

plot.save('colorbar_example.png')
```

### `format_date_axis(ax, date_format='%Y-%m-%d', rotation=0, **kwargs)`

Format date axis labels.

```python
from monet_plots.plot_utils import format_date_axis

# Format date axis
format_date_axis(plot.ax, date_format='%b %Y', rotation=45)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ax` | `matplotlib.axes.Axes` | Required | Axes object to format |
| `date_format` | `str` | `'%Y-%m-%d'` | Date format string |
| `rotation` | `int` | `0` | Label rotation angle |
| `**kwargs` | `dict` | `{}` | Additional formatting parameters |

**Returns:**
- `matplotlib.axes.Axes`: Formatted axes object

**Example:**
```python
import pandas as pd
from monet_plots import TimeSeriesPlot
from monet_plots.plot_utils import format_date_axis

# Create time series plot
plot = TimeSeriesPlot()
dates = pd.date_range('2023-01-01', periods=365, freq='D')
data = pd.DataFrame({
    'date': dates,
    'value': np.random.normal(0, 1, 365)
})

plot.plot(data, x='date', y='value', title='Formatted Date Axis')

# Format date axis
format_date_axis(
    plot.ax,
    date_format='%b %Y',
    rotation=45,
    ha='right'
)

plot.save('formatted_dates.png')
```

### `save_figure(fig, filename, dpi=300, bbox_inches='tight', **kwargs)`

Save a matplotlib figure with consistent parameters.

```python
from monet_plots.plot_utils import save_figure

# Save figure with consistent formatting
save_figure(plot.fig, 'output.png', dpi=600, format='png')
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fig` | `matplotlib.figure.Figure` | Required | Figure to save |
| `filename` | `str` | Required | Output filename |
| `dpi` | `int` | `300` | Resolution in dots per inch |
| `bbox_inches` | `str` | `'tight'` | Bounding box mode |
| `**kwargs` | `dict` | `{}` | Additional save parameters |

**Returns:**
- None

**Example:**
```python
from monet_plots import SpatialPlot
from monet_plots.plot_utils import save_figure

plot = SpatialPlot()
plot.plot(data)
plot.title("High Quality Plot")

# Save with high quality settings
save_figure(
    plot.fig,
    'high_quality_plot.png',
    dpi=600,
    format='png',
    quality=95,
    optimize=True
)
```

### `add_grid(ax, show=True, linestyle=':', alpha=0.5, **kwargs)`

Add grid to axes with consistent styling.

```python
from monet_plots.plot_utils import add_grid

# Add styled grid
add_grid(plot.ax, show=True, linestyle='--', alpha=0.3)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ax` | `matplotlib.axes.Axes` | Required | Axes to add grid to |
| `show` | `bool` | `True` | Whether to show grid |
| `linestyle` | `str` | `':'` | Grid line style |
| `alpha` | `float` | `0.5` | Grid transparency |
| `**kwargs` | `dict` | `{}` | Additional grid parameters |

**Returns:**
- `matplotlib.axes.Axes`: Axes with grid

**Example:**
```python
from monet_plots import ScatterPlot
from monet_plots.plot_utils import add_grid

plot = ScatterPlot()
plot.plot(data, x='x', y='y')

# Add custom grid
add_grid(
    plot.ax,
    show=True,
    linestyle='--',
    alpha=0.3,
    color='gray',
    linewidth=0.5
)

plot.save('grid_example.png')
```

### `create_figure(nrows=1, ncols=1, figsize=(8, 6), **kwargs)`

Create a matplotlib figure with MONET Plots defaults.

```python
from monet_plots.plot_utils import create_figure

# Create figure with defaults
fig, axes = create_figure(nrows=2, ncols=2, figsize=(12, 10))
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nrows` | `int` | `1` | Number of rows |
| `ncols` | `int` | `1` | Number of columns |
| `figsize` | `tuple` | `(8, 6)` | Figure size |
| `**kwargs` | `dict` | `{}` | Additional figure parameters |

**Returns:**
- `tuple`: (figure, axes) objects

**Example:**
```python
from monet_plots.plot_utils import create_figure

# Create multi-panel figure
fig, axes = create_figure(
    nrows=2,
    ncols=2,
    figsize=(14, 10),
    sharex=True,
    sharey=True
)

# Plot on each subplot
for i, ax in enumerate(axes.flat):
    ax.plot(np.random.random(100) + i, label=f'Dataset {i+1}')
    ax.legend()
    ax.set_title(f'Panel {i+1}')

plt.tight_layout()
save_figure(fig, 'multi_panel_plot.png')
```

### `validate_data(data, required_columns=None, data_type=None)`

Validate input data for plotting functions.

```python
from monet_plots.plot_utils import validate_data

# Validate DataFrame
validate_data(df, required_columns=['time', 'value'], data_type='pandas')
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `any` | Required | Data to validate |
| `required_columns` | `list` | `None` | Required column names |
| `data_type` | `str` | `None` | Expected data type |

**Returns:**
- `bool`: True if valid, raises ValueError if invalid

**Example:**
```python
import pandas as pd
from monet_plots.plot_utils import validate_data

# Create valid data
df = pd.DataFrame({
    'time': pd.date_range('2023-01-01', periods=100),
    'value': np.random.normal(0, 1, 100)
})

# Validate data
try:
    validate_data(df, required_columns=['time', 'value'], data_type='pandas')
    print("Data is valid!")
except ValueError as e:
    print(f"Data validation failed: {e}")
```

### `create_subplot_layout(n_plots, max_cols=3, figsize=None, **kwargs)`

Create optimal subplot layout for multiple plots.

```python
from monet_plots.plot_utils import create_subplot_layout

# Create layout for 7 plots
fig, axes = create_subplot_layout(7, max_cols=3, figsize=(15, 10))
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_plots` | `int` | Required | Number of plots to create |
| `max_cols` | `int` | `3` | Maximum number of columns |
| `figsize` | `tuple` | `None` | Figure size (auto-calculated if None) |
| `**kwargs` | `dict` | `{}` | Additional parameters |

**Returns:**
- `tuple`: (figure, axes) objects

**Example:**
```python
from monet_plots.plot_utils import create_subplot_layout

# Create layout for multiple time series
n_plots = 8
fig, axes = create_subplot_layout(n_plots, max_cols=4, figsize=(16, 12))

# Plot data on each subplot
for i, ax in enumerate(axes.flat):
    dates = pd.date_range('2023-01-01', periods=365)
    data = np.random.normal(i, 0.5, 365).cumsum()
    ax.plot(dates, data, label=f'Series {i+1}')
    ax.legend()
    ax.set_title(f'Series {i+1}')

# Format all date axes
for ax in axes.flat:
    format_date_axis(ax, date_format='%b', rotation=45)

plt.tight_layout()
save_figure(fig, 'multi_series_plot.png')
```

### `add_legend(ax, labels=None, location='best', **kwargs)`

Add legend with MONET Plots defaults.

```python
from monet_plots.plot_utils import add_legend

# Add custom legend
add_legend(plot.ax, ['Model A', 'Model B'], location='upper right')
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ax` | `matplotlib.axes.Axes` | Required | Axes to add legend to |
| `labels` | `list` | `None` | Legend labels |
| `location` | `str` | `'best'` | Legend location |
| `**kwargs` | `dict` | `{}` | Additional legend parameters |

**Returns:**
- `matplotlib.legend.Legend`: Legend object

**Example:**
```python
from monet_plots import ScatterPlot
from monet_plots.plot_utils import add_legend

plot = ScatterPlot()

# Plot multiple datasets
plot.ax.scatter(x1, y1, alpha=0.7, label='Dataset 1')
plot.ax.scatter(x2, y2, alpha=0.7, label='Dataset 2')

# Add custom legend
legend = add_legend(
    plot.ax,
    location='upper left',
    fontsize=10,
    framealpha=0.8,
    title='Data Groups'
)

plot.title("Plot with Custom Legend")
plot.save('legend_example.png')
```

### `apply_wiley_style(ax=None, **style_kwargs)`

Apply Wiley style to axes or entire plot.

```python
from monet_plots.plot_utils import apply_wiley_style

# Apply Wiley style to specific axes
apply_wiley_style(plot.ax, fontsize=12)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ax` | `matplotlib.axes.Axes` | `None` | Axes to style (None for all) |
| `**style_kwargs` | `dict` | `{}` | Style parameters |

**Returns:**
- None

**Example:**
```python
from monet_plots import SpatialPlot
from monet_plots.plot_utils import apply_wiley_style

plot = SpatialPlot()

# Apply custom Wiley style
apply_wiley_style(
    plot.ax,
    fontsize=11,
    grid_alpha=0.3,
    linewidth=1.2
)

plot.plot(data)
plot.title("Custom Styled Plot")
plot.save('styled_plot.png')
```

## Helper Classes

### `PlotConfig`

Configuration class for plot settings.

```python
from monet_plots.plot_utils import PlotConfig

# Create plot configuration
config = PlotConfig(
    figsize=(10, 8),
    dpi=300,
    style='wiley',
    grid=True,
    legend=True
)

# Use configuration
plot = SpatialPlot(**config.figure_kwargs)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `figsize` | `tuple` | `(8, 6)` | Figure size |
| `dpi` | `int` | `100` | Resolution |
| `style` | `str` | `'wiley'` | Style name |
| `grid` | `bool` | `True` | Show grid |
| `legend` | `bool` | `False` | Show legend |

**Methods:**
- `to_dict()`: Convert to dictionary
- `update(**kwargs)`: Update configuration

### `DataValidator`

Data validation utility class.

```python
from monet_plots.plot_utils import DataValidator

# Create validator
validator = DataValidator(
    required_columns=['time', 'value'],
    data_type='pandas'
)

# Validate data
try:
    validator.validate(df)
    print("Data is valid for plotting!")
except ValueError as e:
    print(f"Validation error: {e}")
```

## Advanced Usage

### Custom Plot Pipeline

```python
from monet_plots.plot_utils import (
    create_figure, add_grid, apply_wiley_style,
    save_figure, add_legend
)

def create_custom_plot(data, title="", filename="output.png"):
    """Create a custom plot with consistent styling."""

    # Create figure with defaults
    fig, ax = create_figure(figsize=(10, 6))

    # Apply styling
    apply_wiley_style(ax, fontsize=11)

    # Plot data
    ax.plot(data, linewidth=2, label='Data')

    # Add grid and legend
    add_grid(ax, linestyle='--', alpha=0.3)
    add_legend(ax, location='upper right')

    # Add title
    ax.set_title(title, fontsize=14, pad=20)

    # Save figure
    save_figure(fig, filename, dpi=300)

    return fig, ax

# Usage
data = np.random.random(100) + np.arange(100) * 0.1
create_custom_plot(data, title="Custom Plot Example", filename="custom_plot.png")
```

### Batch Processing Multiple Plots

```python
from monet_plots.plot_utils import create_subplot_layout, format_date_axis
import matplotlib.pyplot as plt

def create_batch_plots(data_dict, output_dir="plots/"):
    """Create multiple plots in a batch."""

    n_plots = len(data_dict)
    fig, axes = create_subplot_layout(n_plots, max_cols=2, figsize=(14, 10))

    for (title, data), ax in zip(data_dict.items(), axes.flat):
        ax.plot(data, linewidth=2)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)

    # Format date axes if applicable
    for ax in axes.flat:
        format_date_axis(ax, rotation=45)

    plt.tight_layout()
    save_figure(fig, f"{output_dir}batch_plots.png", dpi=300)

# Usage
data_dict = {
    'Temperature': np.random.normal(20, 5, 365),
    'Humidity': np.random.normal(60, 10, 365),
    'Pressure': np.random.normal(1013, 10, 365),
    'Wind Speed': np.random.gamma(2, 3, 365)
}

create_batch_plots(data_dict)
```

## Performance Tips

1. **Reuse figures**: Use `create_figure()` for consistent figure creation
2. **Validate data early**: Use `validate_data()` to catch issues early
3. **Batch operations**: Use `create_subplot_layout()` for multiple plots
4. **Consistent styling**: Use `apply_wiley_style()` for consistent appearance

## Error Handling

The module includes comprehensive error handling:

```python
try:
    # Validate data before plotting
    validate_data(data, required_columns=['time', 'value'])

    # Create plot
    plot = SpatialPlot()
    plot.plot(data)

    # Save with error handling
    save_figure(plot.fig, 'output.png')

except ValueError as e:
    print(f"Data error: {e}")
except Exception as e:
    print(f"Plotting error: {e}")
```

---

**Related Resources**:

- [Style Configuration](./style) - Plot styling and themes
- [Base API](./base) - Core plotting functionality
- [Examples](../examples) - Practical usage examples
