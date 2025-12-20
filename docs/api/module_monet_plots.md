# Main Package - monet_plots

The `monet_plots` package is the main entry point for the MONET Plots library. It provides convenient imports for all plot classes, style configurations, and utilities.

## Package Overview

```python
import monet_plots

# Check version
print(monet_plots.__version__)
print(monet_plots.__author__)
print(monet_plots.__email__)
```

## Package Contents

### Plot Classes

| Plot Type | Class | Description |
|-----------|-------|-------------|
| Spatial | [`SpatialPlot`](../plots/spatial) | Geospatial plots with cartopy support |
| Time Series | [`TimeSeriesPlot`](../plots/timeseries) | Time series with statistical bands |
| Scatter | [`ScatterPlot`](../plots/scatter) | Scatter plots with regression lines |
| Taylor Diagram | [`TaylorDiagramPlot`](../plots/taylor) | Model evaluation diagrams |
| KDE | [`KDEPlot`](../plots/kde) | Kernel density estimation plots |
| Wind Quiver | [`WindQuiverPlot`](../plots/wind) | Wind vector plots |
| Wind Barbs | [`WindBarbsPlot`](../plots/wind) | Wind barb plots |
| Facet Grid | [`FacetGridPlot`](../plots/facet_grid) | Multi-panel figure layouts |

### Style Configuration

| Component | Description |
|-----------|-------------|
| [`wiley_style`](#wiley_style) | Default Wiley-compliant style |
| [`aps_style`](#aps_style) | APS journal style |
| [`nature_style`](#nature_style) | Nature journal style |
| [`science_style`](#science_style) | Science journal style |
| [`custom_style`](#custom_style) | Custom style creation utility |

### Utilities

| Module | Description |
|--------|-------------|
| [`colorbars`](../colorbars) | Colorbar creation utilities |
| [`taylordiagram`](../taylordiagram) | Taylor diagram functionality |
| [`plot_utils`](../plot_utils) | Common plotting utilities |
| [`cartopy_utils`](../cartopy_utils) | Cartopy integration utilities |
| [`mapgen`](../mapgen) | Map generation utilities |

## Package Attributes

### Version Information

```python
print(monet_plots.__version__)      # Version string
print(monet_plots.__author__)       # Author name
print(monet_plots.__email__)        # Contact email
print(monet_plots.__description__)  # Package description
print(monet_plots.__url__)          # Project URL
```

### Dependencies

```python
# Check required dependencies
print(monet_plots.__dependencies__)
print(monet_plots.__optional_dependencies__)

# Check if optional dependencies are available
print(monet_plots.has_cartopy)      # True if cartopy is installed
print(monet_plots.has_seaborn)     # True if seaborn is installed
print(monet_plots.has_xarray)       # True if xarray is installed
```

## Style Configuration

### `wiley_style`

The default Wiley-compliant style for scientific publications.

```python
import monet_plots
import matplotlib.pyplot as plt

# Apply Wiley style
plt.style.use(monet_plots.wiley_style)
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

American Physical Society (APS) compliant style.

```python
import matplotlib.pyplot as plt
plt.style.use(monet_plots.aps_style)
```

### `nature_style`

Nature journal compliant style.

```python
import matplotlib.pyplot as plt
plt.style.use(monet_plots.nature_style)
```

### `science_style`

Science journal compliant style.

```python
import matplotlib.pyplot as plt
plt.style.use(monet_plots.science_style)
```

### `custom_style(**kwargs)`

Create custom styles by modifying existing ones.

```python
# Create custom style
custom = monet_plots.custom_style(
    font_size=12,
    figure_size=(10, 8),
    grid_style='--'
)

plt.style.use(custom)
```

## Usage Examples

### Basic Import and Usage

```python
import monet_plots
import matplotlib.pyplot as plt
import numpy as np

# Apply default style
plt.style.use(monet_plots.wiley_style)

# Create spatial plot
plot = monet_plots.SpatialPlot(figsize=(10, 8))
data = np.random.random((50, 100))
plot.plot(data, title="Basic Spatial Plot")
plot.save('basic_plot.png')
```

### Working with Different Plot Types

```python
import monet_plots
import pandas as pd
import numpy as np

# Create sample data
dates = pd.date_range('2023-01-01', periods=100, freq='D')
data = pd.DataFrame({
    'time': dates,
    'observed': np.random.normal(0, 1, 100),
    'modeled': np.random.normal(0.1, 1.1, 100)
})

# Create different plot types
spatial_plot = monet_plots.SpatialPlot()
timeseries_plot = monet_plots.TimeSeriesPlot()
scatter_plot = monet_plots.ScatterPlot()

# Plot data
spatial_plot.plot(np.random.random((30, 50)), title="Spatial Data")
timeseries_plot.plot(data, x='time', y='observed', title="Time Series")
scatter_plot.plot(data, x='observed', y='modeled', title="Scatter Plot")

# Save plots
spatial_plot.save('spatial.png')
timeseries_plot.save('timeseries.png')
scatter_plot.save('scatter.png')

# Clean up
spatial_plot.close()
timeseries_plot.close()
scatter_plot.close()
```

### Style Customization

```python
import monet_plots
import matplotlib.pyplot as plt

# Create custom presentation style
presentation_style = monet_plots.custom_style(
    base_style='wiley',
    font_size=14,
    figure_size=(12, 8),
    grid_alpha=0.3
)

plt.style.use(presentation_style)

# Create plots with custom style
plot = monet_plots.SpatialPlot(figsize=(14, 10))
plot.plot(data, title="Presentation Style Plot")
plot.save('presentation_plot.png')
```

### Conditional Import Handling

```python
import monet_plots

# Safe import with optional dependencies
try:
    import cartopy
    print("Cartopy is available for geospatial plotting")
    plot = monet_plots.SpatialPlot()
except ImportError:
    print("Cartopy not available, using basic matplotlib")
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    ax.imshow(np.random.random((50, 100)))
    plt.savefig('fallback_plot.png')
```

## Package Configuration

### Environment Variables

```python
import os
import monet_plots

# Set custom style directory
os.environ['MONET_PLOTS_STYLE_DIR'] = '/path/to/custom/styles'

# Set default figure size
os.environ['MONET_PLOTS_DEFAULT_FIGSIZE'] = '10,8'

# Set output directory
os.environ['MONET_PLOTS_OUTPUT_DIR'] = './plots'
```

### Configuration File

Create a configuration file at `~/.monet_plots_config.json`:

```json
{
  "default_style": "wiley",
  "figure_size": [10, 8],
  "dpi": 300,
  "save_format": "png",
  "grid": true,
  "legend": true,
  "font_size": 12,
  "color_palette": "viridis"
}
```

### Programmatic Configuration

```python
import monet_plots

# Set global configuration
monet_plots.set_config(
    default_style='aps',
    figure_size=(12, 9),
    dpi=300,
    save_format='pdf'
)

# Get current configuration
config = monet_plots.get_config()
print(f"Current DPI: {config['dpi']}")
print(f"Default style: {config['default_style']}")
```

## Advanced Usage

### Plugin System

```python
import monet_plots

# Register custom plot types
class CustomPlot(monet_plots.BasePlot):
    def plot(self, data, **kwargs):
        # Custom plotting logic
        pass

monet_plots.register_plot_type('custom', CustomPlot)

# Use custom plot
plot = monet_plots.create_plot('custom')
plot.plot(data)
```

### Batch Processing

```python
import monet_plots
import glob

# Process multiple files
files = glob.glob('data/*.nc')

for file in files:
    # Load data
    data = load_data(file)

    # Create plot
    plot = monet_plots.SpatialPlot()
    plot.plot(data, title=f"Data from {file}")

    # Save with consistent naming
    filename = file.replace('.nc', '.png').replace('data/', 'plots/')
    plot.save(filename)
    plot.close()
```

### Export and Import

```python
import monet_plots

# Export style configuration
style_config = monet_plots.export_style('wiley')
with open('wiley_style.json', 'w') as f:
    json.dump(style_config, f, indent=2)

# Import style configuration
with open('custom_style.json', 'r') as f:
    custom_style = json.load(f)
monet_plots.import_style('custom', custom_style)
```

## Error Handling

The package includes comprehensive error handling and user-friendly messages:

```python
import monet_plots

try:
    # This will raise ImportError if cartopy is not available
    plot = monet_plots.SpatialPlot()
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install cartopy")
except ValueError as e:
    print(f"Invalid data: {e}")
except Exception as e:
    print(f"Plotting error: {e}")
```

## Performance Optimization

```python
import monet_plots
import matplotlib.pyplot as plt

# Enable performance mode
monet_plots.enable_performance_mode()

# Create plots with optimized settings
plot = monet_plots.SpatialPlot(
    figsize=(8, 6),
    dpi=100,  # Lower DPI for faster rendering
    optimize=True
)

# Disable interactive features for batch processing
plt.ioff()  # Turn off interactive mode

# Process multiple plots efficiently
plots = []
for data in data_list:
    plot = monet_plots.SpatialPlot()
    plot.plot(data)
    plots.append(plot)

# Save all plots at once
monet_plots.batch_save(plots, output_dir='./plots')

# Clean up
monet_plots.cleanup()
```

## Debug Mode

```python
import monet_plots

# Enable debug mode
monet_plots.enable_debug_mode()

# Create plot with debug output
plot = monet_plots.SpatialPlot(debug=True)
plot.plot(data)  # Will show debug information

# Get debug information
debug_info = monet_plots.get_debug_info()
print(f"Memory usage: {debug_info['memory_mb']:.2f} MB")
print(f"Render time: {debug_info['render_time']:.3f} seconds")
```

---

**Related Resources**:

- [Base API](./base) - Core plotting functionality
- [Style Configuration](./style) - Detailed style documentation
- [Plot Types](../plots) - Specific plot implementations
- [Examples](../examples) - Practical usage examples
