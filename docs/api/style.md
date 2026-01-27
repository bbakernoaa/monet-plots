# Style Module

The `monet_plots.style` module provides pre-configured styling options for creating publication-quality plots. All styles are designed to be consistent and professional.

## Overview

MONET Plots includes several built-in styles that can be applied to your plots using matplotlib's style system. These styles ensure consistent appearance across different plot types and publications.

## Available Styles

### `wiley_style`

The default Wiley-compliant style for scientific publications.

```python
from monet_plots import style

# Apply Wiley style
import matplotlib.pyplot as plt
plt.style.use(style.wiley_style)
```

**Style Configuration:**

```python
wiley_style = {
    # Font settings
    'font.family': 'serif',
    'font.serif': 'Times New Roman',
    'font.size': 10,

    # Axes settings
    'axes.labelsize': 10,
    'axes.titlesize': 12,
    'axes.grid': True,
    'grid.linestyle': ':',
    'grid.color': 'gray',

    # Line settings
    'lines.linewidth': 1.5,
    'lines.markersize': 5,

    # Legend settings
    'legend.fontsize': 9,
    'legend.frameon': False,

    # Figure settings
    'figure.figsize': (6, 4),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'tiff',
    'savefig.bbox': 'tight',
}
```

### `aps_style`

American Physical Society (APS) compliant style for physics journals.

```python
import matplotlib.pyplot as plt
plt.style.use(style.aps_style)
```

**Style Configuration:**

```python
aps_style = {
    # Font settings - APS uses Helvetica
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 10,

    # Axes settings
    'axes.labelsize': 10,
    'axes.titlesize': 12,
    'axes.grid': False,

    # Line settings
    'lines.linewidth': 1.5,
    'lines.markersize': 6,

    # Legend settings
    'legend.fontsize': 9,
    'legend.framealpha': 0.8,

    # Figure settings
    'figure.figsize': (5, 4),
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'savefig.format': 'pdf',
}
```

### `nature_style`

Nature journal compliant style.

```python
import matplotlib.pyplot as plt
plt.style.use(style.nature_style)
```

### `science_style`

Science journal compliant style.

```python
import matplotlib.pyplot as plt
plt.style.use(style.science_style)
```

## Custom Style Creation

### `custom_style(**kwargs)`

Create a custom style by modifying an existing style or creating a new one.

```python
from monet_plots import style

# Create custom style from Wiley style
custom = style.custom_style(
    font_size=12,
    figure_size=(10, 8),
    grid_style='--',
    grid_alpha=0.3
)

import matplotlib.pyplot as plt
plt.style.use(custom)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `base_style` | `str` | Base style to modify ('wiley', 'aps', 'nature', 'science') |
| `font_size` | `int` | Base font size |
| `figure_size` | `tuple` | Default figure size (width, height) |
| `font_family` | `str` | Font family ('serif', 'sans-serif', etc.) |
| `grid_style` | `str` | Grid line style ('-', '--', ':', etc.) |
| `grid_alpha` | `float` | Grid transparency (0-1) |
| `save_format` | `str` | Default save format ('png', 'pdf', 'tiff', etc.) |
| `dpi` | `int` | Default DPI for saving |
| `**kwargs` | `dict` | Additional style parameters |

### Example: Creating Custom Styles

```python
from monet_plots import style

# Modify existing style
presentation_style = style.custom_style(
    base_style='wiley',
    font_size=14,
    figure_size=(12, 8),
    grid_alpha=0.5
)

# Create completely new style
dark_style = style.custom_style(
    font_family='sans-serif',
    figure.facecolor='#1a1a1a',
    axes.facecolor='#1a1a1a',
    text.color='white',
    axes.edgecolor='white',
    axes.labelcolor='white',
    xtick.color='white',
    ytick.color='white',
    grid_color='#666666'
)
```

## Style Application

### Global Style Application

```python
import matplotlib.pyplot as plt
from monet_plots import style

# Apply style globally
plt.style.use(style.wiley_style)

# All subsequent plots will use this style
plot1 = SpatialPlot()
plot2 = TimeSeriesPlot()
```

### Context Manager for Temporary Style

```python
import matplotlib.pyplot as plt
from monet_plots import style

# Use style temporarily
with plt.style.context(style.aps_style):
    plot = SpatialPlot()
    plot.plot(data)
    plot.save('aps_style.png')

# Style is automatically reverted
```

### Per-Plot Style Application

```python
from monet_plots import SpatialPlot, style

# Create plot with custom style
plot = SpatialPlot()
plot.plot(data)
plt.style.use(style.nature_style)  # Affects only this plot
plot.save('nature_style.png')
```

## Style Parameters Reference

### Font Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `font.family` | 'serif' | Font family |
| `font.serif` | 'Times New Roman' | Serif font family |
| `font.sans-serif` | ['Arial', 'Helvetica'] | Sans-serif font options |
| `font.monospace` | ['Courier New'] | Monospace font options |
| `font.size` | 10 | Base font size in points |
| `axes.labelsize` | 10 | Axis label font size |
| `axes.titlesize` | 12 | Axis title font size |
| `legend.fontsize` | 9 | Legend font size |
| `xtick.labelsize` | 9 | X-axis tick label size |
| `ytick.labelsize` | 9 | Y-axis tick label size |

### Axes Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `axes.grid` | True | Whether to show grid |
| `grid.linestyle` | ':' | Grid line style |
| `grid.color` | 'gray' | Grid line color |
| `grid.alpha` | 0.5 | Grid transparency |
| `axes.facecolor` | 'white' | Axis background color |
| `axes.edgecolor` | 'black' | Axis edge color |
| `axes.spines.color` | 'black' | Spine color |

### Line and Marker Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lines.linewidth` | 1.5 | Default line width |
| `lines.markersize` | 5 | Default marker size |
| `lines.markeredgewidth` | 1 | Marker edge width |

### Figure Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `figure.figsize` | (6, 4) | Default figure size |
| `figure.dpi` | 100 | Display DPI |
| `savefig.dpi` | 300 | Save DPI |
| `savefig.format` | 'tiff' | Default save format |
| `savefig.bbox` | 'tight' | Save bounding box mode |

### Legend Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `legend.frameon` | False | Whether to show legend frame |
| `legend.framealpha` | 1.0 | Frame transparency |
| `legend.fancybox` | False | Rounded legend frame |
| `legend.shadow` | False | Legend shadow |

## Style Best Practices

### Choosing the Right Style

1. **Publications**: Use journal-specific styles when available
2. **Presentations**: Increase font sizes and figure dimensions
3. **Web**: Use high-contrast styles for better readability
4. **Data Exploration**: Use minimal grid and clear labels

### Customization Guidelines

```python
# Good: Modify existing style incrementally
custom_wiley = style.custom_style(
    base_style='wiley',
    font_size=11,  # Slight modification
    grid_alpha=0.3  # Subtle grid
)

# Avoid: Completely overriding styles
bad_style = {
    'font.size': 12,  # Missing many other parameters
    'figure.figsize': (8, 6)
}
```

### Consistency Across Plots

```python
# Set style at the beginning of your script
import matplotlib.pyplot as plt
from monet_plots import style

plt.style.use(style.wiley_style)

# Create all plots with consistent style
plot1 = SpatialPlot()
plot2 = TimeSeriesPlot()
plot3 = ScatterPlot()
```

## Style Troubleshooting

### Common Issues

**Style Not Applied:**
```python
# Wrong: Style applied after plot creation
plot = SpatialPlot()
plt.style.use(style.wiley_style)  # Too late!
plot.plot(data)

# Correct: Apply style before creating plot
plt.style.use(style.wiley_style)
plot = SpatialPlot()
plot.plot(data)
```

**Missing Fonts:**
```python
# Install missing fonts
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['Arial', 'Helvetica', 'sans-serif']
```

**Style Conflicts:**
```python
# Reset matplotlib style before applying new one
plt.style.use('default')  # Reset to defaults
plt.style.use(style.aps_style)  # Apply new style
```

## Style Templates

### Presentation Templates

```python
# Large format presentation
presentation_style = style.custom_style(
    base_style='wiley',
    font_size=16,
    figure_size=(14, 10),
    grid_alpha=0.3
)

# Small format presentation
small_presentation = style.custom_style(
    base_style='wiley',
    font_size=12,
    figure_size=(8, 6),
    grid_alpha=0.2
)
```

### Publication Templates

```python
# Two-column journal
two_column_style = style.custom_style(
    base_style='wiley',
    figure_size=(3.5, 2.5),  # Two-column width
    font_size=9
)

# Full-page figure
full_page_style = style.custom_style(
    base_style='wiley',
    figure_size=(7, 9),  # Full page
    font_size=10
)
```

---

**Related Resources**:

- [Base Plot API](./base.md) - Core plotting functionality
- [Colorbars API](./colorbars.md) - Colorbar customization
- [Examples](../examples/index.md) - Style usage examples
