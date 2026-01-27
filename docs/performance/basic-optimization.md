# Basic Optimization Techniques

Learn the fundamental optimization techniques that will give you the biggest performance improvements with minimal effort. These strategies are easy to implement and will significantly speed up your plotting workflows.

## Overview

Basic optimization focuses on quick wins that provide immediate performance benefits without requiring complex changes to your code. These techniques are suitable for most users and will solve common performance bottlenecks.

### Key Benefits

- **Faster plot generation** - Reduce wait times from minutes to seconds
- **Lower memory usage** - Handle larger datasets without crashing
- **Better interactivity** - Smoother interactive plotting
- **Improved resource usage** - More efficient CPU and RAM utilization

## Quick Wins

### 1. Downsample Data for Interactive Use

**Problem**: Large datasets make interactive plotting slow and unresponsive.

**Solution**: Downsample data for interactive viewing, use full resolution only for final exports.

```python
import numpy as np
from monet_plots import SpatialPlot

# Before: Slow interactive plotting
large_data = np.random.random((500, 500))  # 250K points
plot = SpatialPlot()
plot.plot(large_data)  # Takes several seconds

# After: Fast interactive plotting
downsampled_data = large_data[::5, ::5]  # 10K points
plot = SpatialPlot()
plot.plot(downsampled_data)  # Takes fractions of a second
```

**Downsampling Strategies**:

```python
import numpy as np

def smart_downsample(data, target_points=10000):
    """Intelligently downsample data to target point count."""
    total_points = np.prod(data.shape)

    if total_points <= target_points:
        return data  # No downsampling needed

    # Calculate downsampling factors
    downsample_factor = int(np.sqrt(total_points / target_points))

    # Apply downsampling
    downsampled = data[::downsample_factor, ::downsample_factor]

    return downsampled

# Usage
data = np.random.random((1000, 1000))
efficient_data = smart_downsample(data, target_points=25000)  # 40x40 grid
```

### 2. Close Plots Properly

**Problem**: Unclosed plots accumulate in memory, causing slowdowns and crashes.

**Solution**: Always close plots when you're done with them.

```python
import matplotlib.pyplot as plt
from monet_plots import TimeSeriesPlot

# Problem: Memory leak
for i in range(10):
    plot = TimeSeriesPlot()
    plot.plot(df, x='time', y='value')
    plot.save(f"plot_{i}.png")
    # plot.close()  # Missing this line causes memory buildup!

# Solution: Proper cleanup
for i in range(10):
    plot = TimeSeriesPlot()
    plot.plot(df, x='time', y='value')
    plot.save(f"plot_{i}.png")
    plot.close()  # Important: Free memory

# Or use context manager for automatic cleanup
for i in range(10):
    with TimeSeriesPlot() as plot:
        plot.plot(df, x='time', y='value')
        plot.save(f"plot_{i}.png")
```

### 3. Use Efficient Data Types

**Problem**: Using inefficient data types slows down operations.

**Solution**: Choose appropriate data types for your data.

```python
import numpy as np
import pandas as pd

# Before: Inefficient data types
data_float64 = np.random.random((1000, 1000)).astype(np.float64)  # 8 bytes per element
memory_usage = data_float64.nbytes / (1024**2)  # ~7.6 MB

# After: Optimized data types
data_float32 = np.random.random((1000, 1000)).astype(np.float32)  # 4 bytes per element
optimized_memory = data_float32.nbytes / (1024**2)  # ~3.8 MB (50% reduction!)

print(f"Memory saved: {memory_usage - optimized_memory:.2f} MB")

# For integer data
data_int64 = np.random.randint(0, 1000, (1000, 1000), dtype=np.int64)  # 8 bytes
data_int32 = np.random.randint(0, 1000, (1000, 1000), dtype=np.int32)  # 4 bytes
```

### 4. Limit Plot Elements

**Problem**: Too many plot elements (markers, lines, text) slow rendering.

**Solution**: Reduce the number of visual elements, especially for large datasets.

```python
from monet_plots import ScatterPlot

# Before: Too many markers
n_points = 100000
x = np.random.normal(0, 1, n_points)
y = np.random.normal(0, 1, n_points)

plot = ScatterPlot()
plot.plot(pd.DataFrame({'x': x, 'y': y}), x='x', y='y')  # Slow with 100K points

# After: Optimized scatter plot
# Sample data for scatter plot
sample_indices = np.random.choice(n_points, size=5000, replace=False)
x_sampled = x[sample_indices]
y_sampled = y[sample_indices]

plot = ScatterPlot()
plot.plot(pd.DataFrame({'x': x_sampled, 'y': y_sampled}), x='x', y='y')  # Much faster
```

## Batch Operations

### Group Similar Operations

**Problem**: Individual plot operations are inefficient when repeated.

**Solution**: Batch similar operations together.

```python
from monet_plots import TimeSeriesPlot
import pandas as pd

# Before: Individual plot creation (slow)
for month in range(1, 13):
    plot = TimeSeriesPlot()
    month_data = df[df['month'] == month]
    plot.plot(month_data, x='date', y='value')
    plot.save(f"month_{month}.png")
    plot.close()

# After: Batch processing (faster)
plots_to_create = []
for month in range(1, 13):
    plot = TimeSeriesPlot()
    month_data = df[df['month'] == month]
    plot.plot(month_data, x='date', y='value')
    plots_to_create.append((plot, f"month_{month}.png"))

# Save all plots at once
for plot, filename in plots_to_create:
    plot.save(filename)
    plot.close()
```

### Pre-computation

**Problem**: Repeated calculations slow down multiple plots.

**Solution**: Pre-compute common calculations.

```python
import numpy as np
from monet_plots import SpatialPlot

# Before: Repeated calculations (slow)
for i in range(5):
    data = np.random.random((100, 100))
    plot = SpatialPlot()
    plot.plot(data, title=f"Plot {i+1}")
    plot.save(f"plot_{i+1}.png")
    plot.close()

# After: Pre-computation (faster)
# Generate all data first
all_data = [np.random.random((100, 100)) for _ in range(5)]

# Create plots with pre-computed data
for i, data in enumerate(all_data):
    plot = SpatialPlot()
    plot.plot(data, title=f"Plot {i+1}")
    plot.save(f"plot_{i+1}.png")
    plot.close()
```

## Efficient Color Mapping

### Use Discrete Colorbars for Large Data

**Problem**: Continuous colorbars with many data points are slow.

**Solution**: Use discrete colorbars for large datasets.

```python
from monet_plots import SpatialPlot
import numpy as np

# Large dataset
large_data = np.random.random((200, 200)) * 100

# Before: Continuous colorbar (slow)
plot = SpatialPlot()
plot.plot(large_data, discrete=False)  # Continuous colorbar

# After: Discrete colorbar (fast)
plot = SpatialPlot()
plot.plot(large_data, discrete=True, ncolors=20)  # Discrete colorbar
```

### Optimize Colormap Selection

**Problem**: Some colormaps are computationally more expensive than others.

**Solution**: Use efficient colormaps.

```python
from monet_plots import SpatialPlot
import numpy as np

data = np.random.random((100, 100))

# Fast colormaps
fast_colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
slow_colormaps = ['jet', 'rainbow', 'gist_rainbow']

# Use fast colormaps for better performance
plot = SpatialPlot()
plot.plot(data, cmap='viridis')  # Fast
plot.save("fast_colormap.png")

# Avoid slow colormaps
# plot = SpatialPlot()
# plot.plot(data, cmap='jet')  # Slow - avoid for large datasets
```

## Interactive Optimization

### Enable Interactive Mode

**Problem**: Plots may not respond well to interactive commands.

**Solution**: Enable interactive mode for better responsiveness.

```python
import matplotlib.pyplot as plt
from monet_plots import TimeSeriesPlot

# Enable interactive mode
plt.ion()

# Create interactive plot
plot = TimeSeriesPlot()
plot.plot(df, x='time', y='value')

# Plot will update immediately when modified
plot.title("Interactive Plot")
plt.draw()  # Force update
```

### Use Aggressive Redrawing

**Problem**: Interactive plots may be slow to update.

**Solution**: Control redrawing behavior.

```python
# Turn off interactive updates during batch operations
plt.ioff()

# Perform multiple operations
plot = TimeSeriesPlot()
plot.plot(df, x='time', y='value')
plot.title("Batch Update")
plot.xlabel("Custom Label")

# Turn interactive updates back on
plt.ion()

# Force a single redraw
plt.draw()
```

## Memory Optimization

### Use Memory-Efficient Data Structures

**Problem**: Large pandas DataFrames consume excessive memory.

**Solution**: Use appropriate data types in pandas.

```python
import pandas as pd
import numpy as np

# Before: Memory-inefficient DataFrame
n_rows = 1_000_000
large_df = pd.DataFrame({
    'id': range(n_rows),
    'value': np.random.random(n_rows),
    'category': np.random.choice(['A', 'B', 'C'], n_rows)
})

print(f"Memory usage: {large_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# After: Memory-optimized DataFrame
optimized_df = pd.DataFrame({
    'id': pd.Series(range(n_rows), dtype='int32'),
    'value': pd.Series(np.random.random(n_rows), dtype='float32'),
    'category': pd.Series(np.random.choice(['A', 'B', 'C'], n_rows), dtype='category')
})

print(f"Optimized memory: {optimized_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
```

### Process Data in Chunks

**Problem**: Loading and processing large datasets at once is memory-intensive.

**Solution**: Process data in manageable chunks.

```python
from monet_plots import SpatialPlot
import numpy as np

# Before: Load all data at once
# huge_data = np.load('huge_dataset.npy')  # May cause memory issues
# plot = SpatialPlot()
# plot.plot(huge_data)  # Memory intensive

# After: Process in chunks
chunk_size = 1000
total_size = 5000

for i in range(0, total_size, chunk_size):
    chunk = np.random.random((chunk_size, chunk_size))
    plot = SpatialPlot()
    plot.plot(chunk, title=f"Chunk {i//chunk_size + 1}")
    plot.save(f"chunk_{i//chunk_size + 1}.png")
    plot.close()
```

## Performance Monitoring

### Simple Timing

```python
import time
from monet_plots import SpatialPlot
import numpy as np

# Simple timing
start_time = time.time()

data = np.random.random((100, 100))
plot = SpatialPlot()
plot.plot(data)
plot.save("timed_plot.png")
plot.close()

end_time = time.time()
print(f"Plot creation took {end_time - start_time:.2f} seconds")
```

### Memory Usage Monitoring

```python
import psutil
import os
from monet_plots import SpatialPlot
import numpy as np

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

# Monitor memory usage
initial_memory = get_memory_usage()
print(f"Initial memory: {initial_memory:.2f} MB")

# Create plot
data = np.random.random((500, 500))
plot = SpatialPlot()
plot.plot(data)
plot.save("monitored_plot.png")
plot.close()

final_memory = get_memory_usage()
print(f"Final memory: {final_memory:.2f} MB")
print(f"Memory used: {final_memory - initial_memory:.2f} MB")
```

## Common Optimization Patterns

### Pattern 1: Lazy Loading

```python
def create_plot_with_lazy_loading(data_source, plot_config):
    """Create plot with lazy data loading."""
    # Load only necessary data
    data = load_data_lazily(data_source)

    # Configure plot
    plot = SpatialPlot(**plot_config)

    # Plot and save
    plot.plot(data)
    plot.save("lazy_loaded_plot.png")
    plot.close()
```

### Pattern 2: Caching

```python
from functools import lru_cache
import numpy as np

@lru_cache(maxsize=10)
def generate_cached_data(shape, seed):
    """Cache generated data to avoid recomputation."""
    np.random.seed(seed)
    return np.random.random(shape)

# Usage
data1 = generate_cached_data((100, 100), 42)  # First call - computes
data2 = generate_cached_data((100, 100), 42)  # Second call - cached
```

### Pattern 3: Batch Processing

```python
def batch_create_plots(data_list, plot_configs):
    """Create multiple plots efficiently."""
    # Pre-process all data
    processed_data = [process_data(data) for data in data_list]

    # Create all plots
    plots = []
    for data, config in zip(processed_data, plot_configs):
        plot = SpatialPlot(**config)
        plot.plot(data)
        plots.append(plot)

    # Save all plots
    for i, plot in enumerate(plots):
        plot.save(f"batch_plot_{i}.png")
        plot.close()
```

## Practice Exercises

### Exercise 1: Data Downsampling
Take a large dataset and implement smart downsampling to achieve a target performance level.

### Exercise 2: Memory Cleanup
Create a script that generates multiple plots and ensures proper memory cleanup.

### Exercise 3: Batch Processing
Implement batch processing for creating time series plots for multiple months.

### Exercise 4: Interactive Optimization
Optimize an interactive plotting workflow for real-time data updates.

### Exercise 5: Performance Monitoring
Add performance monitoring to your existing plotting scripts.

## Troubleshooting

### Issue 1: Still Slow After Optimization

```python
# Profile your code to find bottlenecks
import cProfile
import pstats

def profile_plot_creation():
    data = np.random.random((100, 100))
    plot = SpatialPlot()
    plot.plot(data)
    plot.save("profiled_plot.png")
    plot.close()

# Run profiler
profiler = cProfile.Profile()
profiler.enable()
profile_plot_creation()
profiler.disable()

# Print statistics
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # Show top 10 time-consuming functions
```

### Issue 2: Memory Not Released

```python
# Force garbage collection
import gc

# After creating plots
plot.close()
del plot
gc.collect()  # Force garbage collection
```

### Issue 3: Interactive Plots Unresponsive

```python
# Check interactive mode
import matplotlib.pyplot as plt
print(f"Interactive mode: {plt.isinteractive()}")

# Enable if needed
plt.ion()

# Turn off blitting for complex plots
plot = SpatialPlot()
plot.plot(data)
plot.ax.set_title('Interactive Plot', blit=False)
```

## Next Steps

After mastering basic optimization, explore:

1. **[Memory Management](../memory-management)** - Advanced memory optimization techniques
2. **[Rendering Optimization](../rendering-optimization)** - Fine-tune plot generation speed
3. **[Workflow Optimization](../workflow-optimization)** - Optimize multi-plot workflows
4. **[Benchmarking](../benchmarking)** - Measure and compare performance

## Quick Reference

| Optimization | Technique | Impact |
|--------------|-----------|--------|
| Downsampling | `data[::factor, ::factor]` | High |
| Plot Closing | `plot.close()` | High |
| Data Types | `float32` instead of `float64` | Medium |
| Batch Operations | Group similar operations | Medium |
| Colorbars | Use discrete for large data | Medium |
| Colormaps | Choose efficient colormaps | Low |

---

**Navigation**:

- [Performance Index](../index.md) - All performance guides
- [Memory Management](../memory-management) - Advanced memory optimization
- [Rendering Optimization](../rendering-optimization) - Fast plot generation
- [Workflow Optimization](../workflow-optimization) - Multi-plot workflows
- [Benchmarking](../benchmarking) - Performance measurement
