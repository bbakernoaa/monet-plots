# Basic Plotting with MONET Plots

Welcome to your first tutorial with MONET Plots! This guide will walk you through the fundamental concepts of creating scientific plots using our library.

## Objective

By the end of this tutorial, you will be able to:
- Create different types of basic plots
- Understand the common plotting workflow
- Customize plot appearance
- Save and manage your plots

## Prerequisites

- Basic Python knowledge
- Understanding of numpy and pandas
- MONET Plots installed (`pip install monet_plots`)

## Plotting Workflow

The MONET Plots workflow follows a consistent pattern:

1. **Import** the required classes
2. **Prepare** your data
3. **Initialize** the plot object
4. **Plot** the data
5. **Customize** appearance
6. **Save** the plot
7. **Close** to free memory

## Example 1: Basic Spatial Plot

Let's start with a simple spatial plot:

```python
import numpy as np
from monet_plots import SpatialPlot

# Step 1: Prepare spatial data
# Create a 2D array representing spatial data
data = np.random.random((20, 30)) * 100

# Step 2: Initialize the plot
plot = SpatialPlot(figsize=(10, 8))

# Step 3: Plot the data
# SpatialPlot sets up the map axes. We use standard matplotlib/cartopy methods to plot.
plot.ax.pcolormesh(data)
plot.ax.set_title("Basic Spatial Plot")

# Step 4: Add labels and save
plot.ax.set_xlabel("Longitude")
plot.ax.set_ylabel("Latitude")
plot.save("basic_spatial.png")

# Step 5: Close the plot
plot.close()

print("Basic spatial plot saved successfully!")
```

### Expected Output

You'll see a plot with:
- Random data values displayed as a color map
- Coastlines and borders overlaid
- A colorbar showing the data range
- Title and axis labels

### Key Concepts

- **SpatialPlot**: Specialized for geospatial data with cartopy support
- **figsize**: Controls the plot size (width, height) in inches
- **Automatic styling**: Applied the Wiley-compliant style automatically

## Example 2: Time Series Plot

Now let's create a time series plot:

```python
import pandas as pd
import numpy as np
from monet_plots import TimeSeriesPlot

# Step 1: Prepare time series data
dates = pd.date_range('2023-01-01', periods=100, freq='D')
values = np.cumsum(np.random.normal(0, 1, 100)) + 50  # Random walk

df = pd.DataFrame({
    'time': dates,
    'values': values
})

# Step 2: Initialize the time series plot
plot = TimeSeriesPlot(df=df, figsize=(12, 6))

# Step 3: Plot the data with custom styling
plot.plot(
    x='time',
    y='values',
    title="Time Series Plot",
    ylabel="Temperature (°C)",
    plotargs={'linewidth': 2, 'color': 'blue'},
    fillargs={'alpha': 0.3, 'color': 'lightblue'}
)

# Step 4: Save and display
plot.save("basic_timeseries.png")
plot.close()

print("Time series plot saved successfully!")
```

### Expected Output

The plot will show:
- A time series line with statistical bands
- Properly formatted dates on the x-axis
- Shaded region representing standard deviation
- Professional styling with grid lines

### Key Concepts

- **TimeSeriesPlot**: Handles time-based data with statistical analysis
- **Statistical bands**: Automatically calculates and displays standard deviation
- **Date formatting**: Automatic handling of datetime objects

## Example 3: Scatter Plot with Regression

Let's create a scatter plot with a regression line:

```python
import pandas as pd
import numpy as np
from monet_plots import ScatterPlot

# Step 1: Prepare scatter data
np.random.seed(42)  # For reproducible results
x = np.random.normal(0, 1, 100)
y = 2 * x + np.random.normal(0, 0.5, 100)  # Linear relationship with noise

df = pd.DataFrame({
    'x_variable': x,
    'y_variable': y
})

# Step 2: Initialize the scatter plot
plot = ScatterPlot(df=df, x='x_variable', y='y_variable', title="Scatter Plot with Regression", figsize=(10, 8))

# Step 3: Plot with regression line
plot.plot(
    scatter_kws={'alpha': 0.6, 's': 50},
    line_kws={'linewidth': 2, 'color': 'red'}
)

# Step 4: Add customizations
plot.ax.set_xlabel("X Variable")
plot.ax.set_ylabel("Y Variable")
plot.ax.legend()

# Step 5: Save
plot.save("basic_scatter.png")
plot.close()

print("Scatter plot saved successfully!")
```

### Expected Output

The plot will display:
- Scatter points showing the relationship between X and Y
- A red regression line showing the linear trend
- 95% confidence interval bands
- Legend and properly labeled axes

### Key Concepts

- **ScatterPlot**: Creates scatter plots with statistical regression
- **Regression analysis**: Automatically fits and displays regression line
- **Confidence intervals**: Shows uncertainty in the regression

## Example 4: Taylor Diagram for Model Comparison

Let's create a Taylor diagram for comparing model performance:

```python
import pandas as pd
import numpy as np
from monet_plots import TaylorDiagramPlot

# Step 1: Prepare model data
obs_std = 1.2  # Observation standard deviation
models_data = [
    (1.1, 0.95, 'High Performance'),
    (1.3, 0.88, 'Medium Performance'),
    (0.9, 0.92, 'Low Performance')
]

# Step 2: Initialize Taylor diagram
plot = TaylorDiagramPlot(
    obsstd=obs_std,
    scale=1.8,
    label='Observations'
)

# Step 3: Add model samples
for model_std, correlation, name in models_data:
    # Generate synthetic data for this model
    n_points = 500
    obs = np.random.normal(0, obs_std, n_points)
    model = correlation * obs * (model_std / obs_std) + np.random.normal(0, np.sqrt(model_std**2 * (1 - correlation**2)), n_points)

    df = pd.DataFrame({'obs': obs, 'model': model})
    plot.add_sample(df, col1='obs', col2='model', label=name)

# Step 4: Add contours and finalize
plot.add_contours(levels=[0.5, 0.8, 0.9, 0.95])
plot.finish_plot()

# Step 5: Save
plot.save("basic_taylor.png")
plot.close()

print("Taylor diagram saved successfully!")
```

### Expected Output

The Taylor diagram will show:
- Observation point at the origin
- Model points positioned based on standard deviation and correlation
- Contour lines showing constant correlation values
- Professional styling with proper labels

### Key Concepts

- **TaylorDiagramPlot**: Visualizes model performance statistics
- **Standard deviation**: Radial distance represents variance
- **Correlation**: Angular position represents correlation strength

## Example 5: Wind Vector Plot

Let's create a wind vector plot:

```python
import numpy as np
from monet_plots import WindQuiverPlot

# Step 1: Prepare wind data
lat = np.linspace(30, 50, 12)  # Latitude
lon = np.linspace(-120, -70, 16)  # Longitude
lon_grid, lat_grid = np.meshgrid(lon, lat)

# Create wind field (simple circulation pattern)
u = -np.sin(lat_grid * np.pi / 180) * 5  # Easterly component
v = np.cos(lat_grid * np.pi / 180) * 3   # Northerly component

# Step 2: Initialize wind plot
plot = WindQuiverPlot(figsize=(12, 10))

# Step 3: Plot wind field
plot.plot(
    u, v,
    x=lon_grid,
    y=lat_grid,
    title="Wind Vector Field",
    xlabel="Longitude",
    ylabel="Latitude",
    color=np.sqrt(u**2 + v**2),  # Color by wind speed
    cmap='viridis',
    scale=80,
    scale_units='xy',
    angles='xy'
)

# Step 4: Save
plot.save("basic_wind.png")
plot.close()

print("Wind plot saved successfully!")
```

### Expected Output

The wind plot will display:
- Arrow vectors showing wind direction and magnitude
- Color coding representing wind speed
- Proper geographical coordinates
- Vector field visualization

### Key Concepts

- **WindQuiverPlot**: Specialized for wind vector visualization
- **Vector components**: U (east-west) and V (north-south) components
- **Color mapping**: Wind speed represented by color

## Common Plotting Patterns

### Pattern 1: Data Preparation

```python
# For spatial data
data = np.random.random((20, 30)) * 100
plot = SpatialPlot()
plot.ax.pcolormesh(data)

# For time series data
dates = pd.date_range('2023-01-01', periods=100, freq='D')
values = np.random.normal(0, 1, 100)
df = pd.DataFrame({'time': dates, 'values': values})
plot = TimeSeriesPlot(df=df)
plot.plot(x='time', y='values')

# For scatter data
x = np.random.normal(0, 1, 100)
y = x + np.random.normal(0, 0.5, 100)
df = pd.DataFrame({'x': x, 'y': y})
plot = ScatterPlot(df=df, x='x', y='y')
plot.plot()
```

### Pattern 2: Customization

```python
# Title and labels
plot.ax.set_title("My Custom Title")
plot.ax.set_xlabel("X Axis Label")
plot.ax.set_ylabel("Y Axis Label")

# Styling
# plotargs = {
#     'linewidth': 2,
#     'color': 'blue',
#     'alpha': 0.8
# }
# plot.plot(data, **plotargs)
```

### Pattern 3: Plot Management

```python
# Always remember to close plots when done
plot.save("my_plot.png")
plot.close()

# Or use context manager for automatic cleanup (if supported)
# with SpatialPlot() as plot:
#     plot.ax.pcolormesh(data)
#     plot.save("temp_plot.png")
```

## Troubleshooting Common Issues

### Issue 1: Import Errors

```python
# If you get import errors, ensure all dependencies are installed
# pip install monet_plots matplotlib cartopy seaborn pandas numpy
```

### Issue 2: No Plot Display

```python
# For interactive plotting in notebooks
# %matplotlib inline

# For interactive scripts
import matplotlib.pyplot as plt
plt.ion()
```

### Issue 3: Memory Issues

```python
# Close plots when not in use
plot = SpatialPlot()
plot.ax.pcolormesh(data)
plot.save("output.png")
plot.close()  # Important for memory management
```

## Practice Exercises

### Exercise 1: Modify the Spatial Plot
Change the colormap and add a different title to the spatial plot example.

### Exercise 2: Create Multiple Time Series
Add a second time series to the time series plot and compare them.

### Exercise 3: Customize Scatter Plot
Change the marker size, color, and add a different regression line style.

### Exercise 4: Add More Models to Taylor Diagram
Add two more models with different performance characteristics.

### Exercise 5: Create Wind Animation
Modify the wind plot to create a simple animated wind field.

## Next Steps

After completing this basic tutorial, you're ready to explore:

1. **[Data Preparation](../getting-started/data-preparation)** - Learn about different data formats
2. **[Customization](../getting-started/plot-customization)** - Advanced styling options
3. **[Core Plot Types](../core-plots)** - Deep dive into individual plot types
4. **[Multi-Plot Layouts](../advanced-workflows/multi-plot-layouts)** - Combine multiple plots

## Quick Reference

| Plot Type | Class | Best For |
|-----------|-------|----------|
| Spatial | `SpatialPlot` | Geospatial data with maps |
| Time Series | `TimeSeriesPlot` | Time-based data with statistics |
| Scatter | `ScatterPlot` | Variable relationships with regression |
| Taylor Diagram | `TaylorDiagramPlot` | Model performance comparison |
| Wind Vectors | `WindQuiverPlot` | Meteorological wind data |

---

**Navigation**:

- [Examples Index](../index.md) - All examples and tutorials
- [Getting Started Guide](../../getting-started.md) - Installation and setup
- [API Reference](../../api/index.md) - Complete API documentation
- [Plot Types](../../plots/index.md) - Individual plot type documentation
