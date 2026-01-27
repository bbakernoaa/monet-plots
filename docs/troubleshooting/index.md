# Troubleshooting and FAQ

This comprehensive troubleshooting guide addresses common issues you may encounter while using MONET Plots, with solutions and best practices to ensure smooth plotting experiences.

## Common Issues and Solutions

### Installation and Setup Issues

#### ImportError: No module named 'monet_plots'

**Problem**: Cannot import MONET Plots after installation.

**Solutions**:

```bash
# 1. Check if package is installed
pip list | grep monet_plots

# 2. If not installed, install it
pip install monet_plots

# 3. If installed, try reinstalling
pip uninstall monet_plots
pip install monet_plots

# 4. Check Python environment
which python  # Should show your virtual environment if active
python -c "import sys; print(sys.executable)"
```

**Python Code Solution**:
```python
# Verify installation
try:
    import monet_plots
    print(f"MONET Plots version: {monet_plots.__version__}")
    print("Installation successful!")
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install monet_plots: pip install monet_plots")
```

#### Missing Optional Dependencies

**Problem**: Error when trying to use cartopy or other optional features.

**Solutions**:

```bash
# Install specific optional dependencies
pip install cartopy  # For geospatial plotting
pip install xarray netcdf4  # For NetCDF support
pip install seaborn pandas  # For enhanced plotting
pip install scipy  # For statistical functions
```

**Code Solution**:
```python
# Check for optional dependencies
optional_deps = {
    'cartopy': 'Geospatial plotting',
    'xarray': 'NetCDF data handling',
    'seaborn': 'Enhanced statistical plotting',
    'scipy': 'Advanced statistical functions'
}

missing_deps = []
for dep, description in optional_deps.items():
    try:
        __import__(dep)
        print(f"✓ {dep} - {description}")
    except ImportError:
        print(f"✗ {dep} - {description} (not installed)")
        missing_deps.append(dep)

if missing_deps:
    print(f"\nInstall missing dependencies: pip install {' '.join(missing_deps)}")
```

#### Version Compatibility Issues

**Problem**: MONET Plots conflicts with other packages due to version requirements.

**Solutions**:

```bash
# Check package versions
pip show matplotlib pandas numpy

# Install compatible versions
pip install matplotlib>=3.3.0 pandas>=1.0.0 numpy>=1.18.0
pip install monet_plots

# Or use conda for better dependency management
conda install -c conda-forge monet_plots matplotlib pandas numpy
```

### Plot Creation Issues

#### Empty Plots or No Data Displayed

**Problem**: Plots are created but show no data or appear blank.

**Common Causes and Solutions**:

```python
import numpy as np
import pandas as pd
from monet_plots import SpatialPlot

# Cause 1: Data format issues
# Solution: Check data shape and type
data = np.random.random((10, 10))  # Correct 2D array
print(f"Data shape: {data.shape}, dtype: {data.dtype}")

# Cause 2: Missing or invalid data
# Solution: Handle missing values
data_with_nan = np.random.random((10, 10))
data_with_nan[5:8, 3:6] = np.nan  # Add missing values

# Use masked arrays for better visualization
masked_data = np.ma.masked_invalid(data_with_nan)

plot = SpatialPlot()
plot.plot(masked_data, title="Data with Missing Values")
plot.save("masked_data_plot.png")
plot.close()

# Cause 3: Incorrect data dimensions
# Solution: Ensure data matches plot requirements
try:
    plot = SpatialPlot()
    plot.plot(np.random.random(100))  # 1D array - will fail
    plot.save("invalid_plot.png")
except ValueError as e:
    print(f"Error: {e}")
    print("Spatial plots require 2D arrays")

# Correct usage
plot = SpatialPlot()
plot.plot(np.random.random((10, 10)))  # 2D array - works
plot.save("valid_plot.png")
plot.close()
```

#### Memory Errors with Large Datasets

**Problem**: Out of memory errors when plotting large datasets.

**Solutions**:

```python
import numpy as np
from monet_plots import SpatialPlot

# Strategy 1: Downsample data
large_data = np.random.random((5000, 5000))  # 25M points

# Simple downsampling
downsampled = large_data[::10, ::10]  # 250K points (100x reduction)

plot = SpatialPlot()
plot.plot(downsampled, title="Downsampled Data")
plot.save("downsampled_plot.png")
plot.close()

# Strategy 2: Process in chunks
def process_large_data_in_chunks(data, chunk_size=1000):
    """Process large spatial data in chunks."""
    h, w = data.shape
    for i in range(0, h, chunk_size):
        for j in range(0, w, chunk_size):
            chunk = data[i:i+chunk_size, j:j+chunk_size]
            # Process and save chunk
            plot = SpatialPlot()
            plot.plot(chunk)
            plot.save(f"chunk_{i}_{j}.png")
            plot.close()

# Process a section of the large data
process_large_data_in_chunks(large_data[:2000, :2000])

# Strategy 3: Use more efficient data types
efficient_data = large_data.astype(np.float32)  # 50% memory reduction
print(f"Original: {large_data.nbytes / 1024**2:.1f} MB")
print(f"Optimized: {efficient_data.nbytes / 1024**2:.1f} MB")
```

#### Slow Plot Rendering

**Problem**: Plots take a long time to render, especially with large datasets.

**Solutions**:

```python
import numpy as np
import time
from monet_plots import SpatialPlot

# Performance optimization techniques
data = np.random.random((2000, 2000))

# Technique 1: Use discrete colorbars for large data
start_time = time.time()
plot = SpatialPlot()
plot.plot(data, discrete=True, ncolors=20)
plot.save("discrete_colormap_plot.png")
plot.close()
discrete_time = time.time() - start_time
print(f"Discrete colormap: {discrete_time:.2f}s")

# Technique 2: Choose efficient colormaps
fast_colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']

for cmap in fast_colormaps:
    start_time = time.time()
    plot = SpatialPlot()
    plot.plot(data, cmap=cmap)
    plot.save(f"fast_cmap_{cmap}.png")
    plot.close()
    print(f"{cmap}: {time.time() - start_time:.2f}s")

# Technique 3: Reduce resolution for previews
low_res_data = data[::4, ::4]  # 16x reduction
start_time = time.time()
plot = SpatialPlot()
plot.plot(low_res_data)
plot.save("low_res_preview.png")
plot.close()
print(f"Low res preview: {time.time() - start_time:.2f}s")
```

#### Colorbar Issues

**Problem**: Colorbars not displaying correctly or not aligning with plots.

**Solutions**:

```python
from monet_plots import SpatialPlot, colorbar_index
import numpy as np

# Problem: Colorbar not matching data
data = np.random.random((50, 50)) * 100

# Solution 1: Use discrete colorbars
plot = SpatialPlot()
plot.plot(data, discrete=True, ncolors=15)
plot.save("discrete_colorbar.png")
plot.close()

# Solution 2: Custom colorbar creation
plot = SpatialPlot()
plot.plot(data, cmap='viridis')

# Add custom colorbar
cbar, cmap = colorbar_index(15, 'viridis', minval=0, maxval=100, dtype=int)
plot.save("custom_colorbar.png")
plot.close()

# Solution 3: Handle data ranges properly
data_with_extreme_values = np.random.random((50, 50))
data_with_extreme_values[10:15, 20:25] = 1000  # Extreme values

plot = SpatialPlot()
plot.plot(
    data_with_extreme_values,
    vmin=0,
    vmax=100,
    title="Data with Extreme Values (Clamped)"
)
plot.save("clamped_colorbar.png")
plot.close()
```

#### Styling and Formatting Issues

**Problem**: Plots don't match expected styling or formatting.

**Solutions**:

```python
import matplotlib.pyplot as plt
from monet_plots import SpatialPlot, style
import numpy as np

# Problem: Default styling not applied
# Solution: Apply MONET Plots style
plt.style.use(style.wiley_style)

data = np.random.random((50, 50))
plot = SpatialPlot()
plot.plot(data, title="Wiley Style Applied")
plot.save("styled_plot.png")
plot.close()

# Problem: Font sizes too small or too large
# Solution: Customize font settings
custom_style = {
    'font.size': 12,
    'axes.labelsize': 10,
    'axes.titlesize': 14,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 10
}

plt.style.use(custom_style)

plot = SpatialPlot()
plot.plot(data, title="Custom Font Sizes")
plot.save("custom_fonts.png")
plot.close()

# Problem: Labels not showing
# Solution: Ensure labels are properly set
plot = SpatialPlot()
plot.plot(data, title="Proper Labels")
plot.xlabel("Longitude (degrees)")
plot.ylabel("Latitude (degrees)")
plot.save("proper_labels.png")
plot.close()
```

### Data Handling Issues

#### Time Series Data Issues

**Problem**: Time series plots not displaying correctly with datetime data.

**Solutions**:

```python
import pandas as pd
import numpy as np
from monet_plots import TimeSeriesPlot

# Problem: Incorrect datetime format
# Solution: Ensure proper datetime handling

# Create proper datetime data
dates = pd.date_range('2023-01-01', periods=100, freq='D')
values = np.cumsum(np.random.normal(0, 1, 100))

df = pd.DataFrame({
    'date': dates,
    'value': values,
    'category': np.random.choice(['A', 'B'], 100)
})

# Correct time series plotting
plot = TimeSeriesPlot()
plot.plot(df, x='date', y='value', title="Proper Time Series")
plot.save("correct_timeseries.png")
plot.close()

# Problem: Missing datetime values
# Solution: Handle missing data
df_with_missing = df.copy()
df_with_missing.loc[10:15, 'value'] = np.nan

plot = TimeSeriesPlot()
plot.plot(df_with_missing, x='date', y='value', title="Time Series with Missing Data")
plot.save("timeseries_missing_data.png")
plot.close()

# Problem: Multiple time series not distinguished
# Solution: Use proper coloring and legends
plot = TimeSeriesPlot()
for category in ['A', 'B']:
    subset = df[df['category'] == category]
    plot.plot(subset, x='date', y='value', label=category)

plot.title("Multiple Time Series")
plot.legend()
plot.save("multiple_timeseries.png")
plot.close()
```

#### Pandas DataFrame Issues

**Problem**: DataFrames not compatible with MONET Plots functions.

**Solutions**:

```python
import pandas as pd
import numpy as np
from monet_plots import ScatterPlot

# Problem: Column names not matching
# Solution: Check and rename columns

# Create DataFrame with problematic column names
df = pd.DataFrame({
    'x_values': np.random.normal(0, 1, 100),
    'y_values': np.random.normal(0, 1, 100),
    'group_col': np.random.choice(['Group 1', 'Group 2'], 100)
})

# Rename columns to match expected format
df_renamed = df.rename(columns={
    'x_values': 'x',
    'y_values': 'y',
    'group_col': 'group'
})

# Check DataFrame structure
print("Original columns:", list(df.columns))
print("Renamed columns:", list(df_renamed.columns))

# Plot with renamed columns
plot = ScatterPlot()
plot.plot(df_renamed, x='x', y='y', title="Correct Column Names")
plot.save("correct_columns.png")
plot.close()

# Problem: Non-numeric data causing issues
# Solution: Handle categorical data properly
categorical_df = pd.DataFrame({
    'measurement': np.random.normal(0, 1, 100),
    'category': np.random.choice(['Type A', 'Type B', 'Type C'], 100),
    'quality': np.random.choice(['High', 'Low'], 100)
})

# Convert categorical to numeric if needed
categorical_df['category_numeric'] = pd.Categorical(categorical_df['category']).codes

plot = ScatterPlot()
plot.plot(categorical_df, x='measurement', y='category_numeric',
plot.plot(categorical_df, x='measurement', y='category_numeric',
         title="Categorical Data Handling")
plot.save("categorical_handling.png")
plot.close()
```

### Performance Issues

#### Memory Leaks

**Problem**: Memory usage increases over time when creating multiple plots.

**Solutions**:

```python
import psutil
import os
import numpy as np
from monet_plots import SpatialPlot

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

# Problem: Memory buildup from unclosed plots
initial_memory = get_memory_usage()
print(f"Initial memory: {initial_memory:.2f} MB")

# Bad practice: Not closing plots
for i in range(10):
    data = np.random.random((100, 100))
    plot = SpatialPlot()
    plot.plot(data)
    plot.save(f"memory_leak_plot_{i}.png")
    # plot.close()  # This causes memory buildup!

memory_after_leak = get_memory_usage()
print(f"Memory after leak: {memory_after_leak:.2f} MB")
print(f"Memory wasted: {memory_after_leak - initial_memory:.2f} MB")

# Good practice: Always close plots
for i in range(10):
    data = np.random.random((100, 100))
    plot = SpatialPlot()
    plot.plot(data)
    plot.save(f"proper_cleanup_plot_{i}.png")
    plot.close()  # Proper cleanup

memory_after_cleanup = get_memory_usage()
print(f"Memory after cleanup: {memory_after_cleanup:.2f} MB")
print(f"Memory used per plot: {(memory_after_cleanup - initial_memory) / 10:.2f} MB")

# Best practice: Use context managers
for i in range(10):
    data = np.random.random((100, 100))
    with SpatialPlot() as plot:
        plot.plot(data)
        plot.save(f"context_manager_plot_{i}.png")
```

#### Slow Interactive Plotting

**Problem**: Interactive plots are unresponsive or slow to update.

**Solutions**:

```python
import matplotlib.pyplot as plt
import numpy as np
from monet_plots import TimeSeriesPlot

# Enable interactive mode
plt.ion()

# Problem: Slow interactive updates
# Solution: Downsample data for interactive use

# Large dataset (slow for interactive)
interactive_data = np.cumsum(np.random.normal(0, 1, 10000))
dates = pd.date_range('2023-01-01', periods=10000, freq='H')
large_df = pd.DataFrame({'time': dates, 'value': interactive_data})

# Downsample for interactive plotting
df_interactive = large_df.iloc[::10, :].copy()  # 10x reduction

plot = TimeSeriesPlot()
plot.plot(df_interactive, x='time', y='value', title="Interactive (Downsampled)")
plt.draw()  # Force update

# Problem: Interactive updates too frequent
# Solution: Control update frequency
plt.ioff()  # Turn off interactive updates temporarily

# Make multiple changes
plot.title Updated Title")
plot.ylabel("Updated Y-axis")
plt.xlabel("Updated X-axis")

plt.ion()  # Turn interactive updates back on
plt.draw()  # Single update

# Problem: Interactive mode not working
# Solution: Check and force interactive mode
print(f"Interactive mode: {plt.isinteractive()}")

if not plt.isinteractive():
    plt.ion()
    print("Interactive mode enabled")


    # Test interactive update
    plot.title("Interactive Test")
    plt.draw()
```

## Advanced Troubleshooting

### Debug Information Collection

**Problem**: Need detailed information about the current state for debugging.

**Solutions**:

```python
import sys
import platform
import matplotlib
import pandas as pd
import numpy as np
from monet_plots import SpatialPlot

def collect_system_info():
    """Collect comprehensive system information."""
    info = {
        'System': platform.system(),
        'Python Version': sys.version,
        'Matplotlib Version': matplotlib.__version__,
        'Pandas Version': pd.__version__,
        'NumPy Version': np.__version__,
        'Platform': platform.platform(),
        'Processor': platform.processor(),
        'Memory': f"{round(platform.meminfo().total / (1024**3), 2)} GB"
    }


    print("System Information:")
    for key, value in info.items():
        print(f"{key}: {value}")


    return info

def collect_plot_debug_info():
    """Collect plot-specific debug information."""
    try:
        # Test basic plot creation
        test_data = np.random.random((10, 10))


        plot = SpatialPlot()
        plot.plot(test_data)


        debug_info = {
            'Plot Created': True,
            'Data Shape': test_data.shape,
            'Data Type': test_data.dtype,
            'Memory Usage': f"{test_data.nbytes / 1024} KB",
            'Matplotlib Backend': matplotlib.get_backend(),
            'Figure Size': plot.fig.get_size_inches(),
            'DPI': plot.fig.get_dpi()
        }


        print("\nPlot Debug Information:")
        for key, value in debug_info.items():
            print(f"{key}: {value}")


        plot.close()
        return debug_info


    except Exception as e:
        print(f"\nPlot Creation Failed: {e}")
        return {'Plot Created': False, 'Error': str(e)}

# Collect and display debug information
system_info = collect_system_info()
plot_info = collect_plot_debug_info()
```

### Performance Profiling

**Problem**: Need to identify performance bottlenecks in plotting code.

**Solutions**:

```python
import cProfile
import pstats
import io
import numpy as np
from monet_plots import SpatialPlot, TimeSeriesPlot, ScatterPlot

def profile_plot_operations():
    """Profile different plot operations."""


    def profile_spatial_plot():
        data = np.random.random((500, 500))
        plot = SpatialPlot()
        plot.plot(data)
        plot.save("profiled_spatial.png")
        plot.close()


    def profile_timeseries_plot():
        import pandas as pd
        dates = pd.date_range('2023-01-01', periods=1000, freq='D')
        values = np.cumsum(np.random.normal(0, 1, 1000))
        df = pd.DataFrame({'time': dates, 'value': values})


        plot = TimeSeriesPlot()
        plot.plot(df, x='time', y='value')
        plot.save("profiled_timeseries.png")
        plot.close()


    def profile_scatter_plot():
        import pandas as pd
        x = np.random.normal(0, 1, 5000)
        y = x * 2 + np.random.normal(0, 1, 5000)
        df = pd.DataFrame({'x': x, 'y': y})


        plot = ScatterPlot()
        plot.plot(df, x='x', y='y')
        plot.save("profiled_scatter.png")
        plot.close()


    # Profile each operation
    operations = [
        ("Spatial Plot", profile_spatial_plot),
        ("Time Series Plot", profile_timeseries_plot),
        ("Scatter Plot", profile_scatter_plot)
    ]


    for name, operation in operations:
        print(f"\nProfiling {name}:")


        profiler = cProfile.Profile()
        profiler.enable()
        operation()
        profiler.disable()


        # Get statistics
        stats_stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stats_stream)
        stats.sort_stats('cumulative')
        stats.print_stats(5)  # Top 5 functions


        print(stats_stream.getvalue())
```

### Error Logging and Recovery

**Problem**: Need robust error handling and logging for production plotting.

**Solutions**:

```python
import logging
import traceback
from datetime import datetime
from monet_plots import SpatialPlot
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('plotting_errors.log'),
        logging.StreamHandler()
    ]
)

def safe_plot_creation(data, config, output_path):
    """Safely create plots with error handling and logging."""
    try:
        logging.info(f"Starting plot creation for {output_path}")
        logging.info(f"Data shape: {data.shape}, config: {config}")


        # Validate input data
        if data is None:
            raise ValueError("Data cannot be None")
        if not hasattr(data, 'shape'):
            raise ValueError("Data must be a numpy array or similar")


        # Create plot
        plot = SpatialPlot(**config.get('plot_kwargs', {}))


        # Plot data
        plot_kwargs = config.get('plot_kwargs', {})
        plot.plot(data, **plot_kwargs)


        # Save plot
        plot.save(output_path, **config.get('save_kwargs', {}))
        plot.close()


        logging.info(f"Successfully created plot: {output_path}")
        return True


    except MemoryError as e:
        logging.error(f"Memory error creating {output_path}: {e}")
        # Try with downsampled data
        try:
            downsampled_data = data[::2, ::2]
            logging.info(f"Attempting with downsampled data: {downsampled_data.shape}")
            return safe_plot_creation(downsampled_data, config, f"downsampled_{output_path}")
        except Exception as retry_error:
            logging.error(f"Retry failed for {output_path}: {retry_error}")
            return False


    except Exception as e:
        logging.error(f"Error creating {output_path}: {e}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        return False

# Usage example
plot_config = {
    'plot_kwargs': {
        'title': 'Safe Plot Creation Example',
        'cmap': 'viridis'
    },
    'save_kwargs': {
        'dpi': 300
    }
}

test_data = np.random.random((1000, 1000))
success = safe_plot_creation(test_data, plot_config, "safe_plot.png")

if success:
    print("Plot created successfully!")
else:
    print("Plot creation failed. Check logs for details.")
```

## FAQ (Frequently Asked Questions)

### General Questions

**Q1: What is the best way to learn MONET Plots?**

**A**: Start with the [Getting Started Guide](../getting-started.md), then explore the [Examples](../examples/index.md) section. Begin with basic plots and gradually work up to more complex visualizations.

**Q2: Can I use MONET Plots with Jupyter notebooks?**

**A**: Yes! MONET Plots works seamlessly with Jupyter. Use `%matplotlib inline` for static plots or `%matplotlib widget` for interactive plots.

```python
%matplotlib inline
from monet_plots import SpatialPlot
plot = SpatialPlot()
plot.plot(data)
plot.save("notebook_plot.png")
```

**Q3: How do I contribute to MONET Plots?**

**A**: We welcome contributions! Please see our [contributing guidelines](../CONTRIBUTING.md) for details on reporting issues, suggesting features, or submitting code.

### Technical Questions

**Q4: What are the system requirements for MONET Plots?**

**A**:
**A**:
- Python 3.7+
- 512MB RAM minimum (1GB+ recommended)
- 50MB disk space
- Optional: cartopy for geospatial features, xarray for NetCDF support

**Q5: Can I use MONET Plots with data from NetCDF files?**

**A**: Yes, if you have xarray installed:

```python
import xarray as xr
from monet_plots import SpatialPlot

ds = xr.open_dataset('data.nc')
plot = SpatialPlot()
plot.plot(ds.temperature)
plot.save("netcdf_plot.png")
plot.close()
```

**Q6: How do I handle missing data in plots?**

**A**: MONET Plots automatically handles NaN values, but you can also use masked arrays for better control:

```python
import numpy as np
from monet_plots import SpatialPlot

data = np.random.random((50, 50))
data[10:20, 10:20] = np.nan  # Add missing values

masked_data = np.ma.masked_invalid(data)
plot = SpatialPlot()
plot.plot(masked_data)
plot.save("missing_data_plot.png")
plot.close()
```

### Performance Questions

**Q7: My plots are very slow, how can I improve performance?**

**A**: Try these optimizations:
1. Downsample data: `data[::4, ::4]` (16x reduction)
2. Use discrete colorbars: `discrete=True, ncolors=20`
3. Close plots properly: `plot.close()`
4. Use efficient data types: `data.astype(np.float32)`

**Q8: How much memory do large plots use?**

**A**: Memory usage depends on data size and type:
- Float64: 8 bytes per element
- Float32: 4 bytes per element (50% reduction)
- Int32: 4 bytes per element

Example: 1000x1000 float64 array = ~7.6MB

**Q9: Can I create batch plots efficiently?**

**A**: Yes! Use batch processing to create multiple plots efficiently:

```python
from monet_plots import SpatialPlot
import numpy as np

# Generate all data first
datasets = [np.random.random((100, 100)) for _ in range(10)]

# Create plots in batch
plots = []
for i, data in enumerate(datasets):
    plot = SpatialPlot()
    plot.plot(data, title=f"Plot {i+1}")
    plots.append(plot)

# Save all plots
for i, plot in enumerate(plots):
    plot.save(f"batch_plot_{i+1}.png")
    plot.close()
```

### Styling Questions

**Q10: How do I create publication-quality plots?**

**A**: Use the built-in Wiley-compliant style:

```python
import matplotlib.pyplot as plt
from monet_plots import style, TimeSeriesPlot

plt.style.use(style.wiley_style)

plot = TimeSeriesPlot()
plot.plot(data, title="Publication-Quality Plot")
plot.xlabel("X Axis Label")
plot.ylabel("Y Axis Label")
plot.save("publication_plot.png", dpi=300)
plot.close()
```

**Q11: Can I customize the appearance of plots?**

**A**: Yes, you can customize almost every aspect:

```python
from monet_plots import SpatialPlot

plot = SpatialPlot(
    figsize=(12, 8),
    dpi=150
)

plot.plot(
    data,
    cmap='viridis',
    title="Custom Styled Plot",
    fontsize=12,
    linewidth=2
)

plot.xlabel("Custom X Label", fontsize=10)
plot.ylabel("Custom Y Label", fontsize=10)
plot.save("custom_styled_plot.png")
plot.close()
```

**Q12: How do I create consistent styling across multiple plots?**

**A**: Define a custom style and reuse it:

```python
import matplotlib.pyplot as plt
from monet_plots import style

# Create custom style
custom_style = {
    'font.size': 12,
    'axes.labelsize': 10,
    'axes.titlesize': 14,
    'figure.figsize': (10, 6),
    'savefig.dpi': 300
}

# Save style
plt.style.use(custom_style)

# Apply to all plots
from monet_plots import SpatialPlot, TimeSeriesPlot

plot1 = SpatialPlot()
plot1.plot(data1, title="Consistent Style 1")
plot1.save("consistent_plot1.png")

plot2 = TimeSeriesPlot()
plot2.plot(data2, title="Consistent Style 2")
plot2.save("consistent_plot2.png")
```

## Getting Help

### Community Support

- **GitHub Issues**: Report bugs and request features
- **GitHub Discussions**: Ask questions and share knowledge
- **Documentation**: Check the full [documentation](../index.md)

### Professional Support

For professional support and consulting:
- Contact the development team
- Check for commercial support options
- Consider training and workshops

### Contributing to Documentation

Help improve this troubleshooting guide by:
- Adding new issues and solutions
- Improving existing explanations
- Providing code examples
- Reporting unclear sections

---

**Related Resources**:

- [Getting Started Guide](../getting-started.md) - Installation and basic usage
- [API Reference](../api/index.md) - Complete API documentation
- [Examples](../examples/index.md) - Practical examples and tutorials
- [Performance Guide](../performance/index.md) - Optimization techniques
- [Configuration Guide](../configuration/index.md) - Customization options
