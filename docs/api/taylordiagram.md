# Taylor Diagram Module

The `monet_plots.taylordiagram` module provides functionality for creating Taylor diagrams, which are commonly used in meteorology and climate science to evaluate model performance against observations.

## Overview

Taylor diagrams display the standard deviation and correlation coefficient between model predictions and observations, providing a comprehensive view of model performance in a single plot.

## TaylorDiagram Class

```python
class TaylorDiagram:
    """Create a Taylor diagram for model evaluation."""


    def __init__(self, obsstd, scale=1.5, fig=None, rect=111, label='OBS', **kwargs):
        """Initialize a Taylor diagram.


        Args:
            obsstd (float): Standard deviation of observations
            scale (float): Scale factor for the diagram
            fig (matplotlib.figure.Figure, optional): Figure to use
            rect (tuple or int): Position and size of subplot
            label (str): Label for observations
            **kwargs: Additional parameters
        """
        pass
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `obsstd` | `float` | Required | Standard deviation of observations |
| `scale` | `float` | `1.5` | Scale factor for diagram size |
| `fig` | `matplotlib.figure.Figure` | `None` | Existing figure to use |
| `rect` | `tuple` or `int` | `111` | Subplot position and size |
| `label` | `str` | `'OBS'` | Label for observations |
| `**kwargs` | `dict` | `{}` | Additional parameters |

### Public Methods

#### `add_sample(stddev, corrcoef, marker='o', label='', **kwargs)`

Add a model sample to the Taylor diagram.

```python
td = TaylorDiagram(obsstd=1.0)
td.add_sample(stddev=0.8, corrcoef=0.9, marker='s', label='Model 1')
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `stddev` | `float` | Required | Standard deviation of the model |
| `corrcoef` | `float` | Required | Correlation coefficient |
| `marker` | `str` | `'o'` | Marker style |
| `label` | `str` | `''` | Model label |
| `**kwargs` | `dict` | `{}` | Additional plot parameters |

**Example:**
```python
from monet_plots import taylordiagram

# Create Taylor diagram
td = taylordiagram.TaylorDiagram(obsstd=1.2, scale=2.0, label='Observations')

# Add model samples
td.add_sample(stddev=1.1, corrcoef=0.95, marker='o', label='Model A', color='red')
td.add_sample(stddev=1.3, corrcoef=0.88, marker='s', label='Model B', color='blue')
td.add_sample(stddev=0.9, corrcoef=0.92, marker='^', label='Model C', color='green')

# Add contours and legend
td.add_contours(levels=[0.5, 0.8, 0.9, 0.95])
td.finish_plot()
```

#### `add_contours(levels=None, **kwargs)`

Add contour lines to the Taylor diagram.

```python
td.add_contours(levels=[0.5, 0.8, 0.9, 0.95], colors='gray', linestyles='--')
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `levels` | `list` | `None` | Contour levels (correlation coefficients) |
| `**kwargs` | `dict` | `{}` | Additional contour parameters |

**Example:**
```python
# Add custom contours
td.add_contours(
    levels=[0.6, 0.7, 0.8, 0.9, 0.95],
    colors='gray',
    linestyles=['-', '--', ':', '-.', (0, (3, 1, 1, 1))],
    linewidths=1.5,
    alpha=0.7
)
```

#### `finish_plot()`

Finalize the plot by adding legend and adjusting layout.

```python
td.finish_plot()
td.save('taylor_diagram.png')
```

**Parameters:**
- None

**Example:**
```python
td = taylordiagram.TaylorDiagram(obsstd=1.0)
td.add_sample(0.8, 0.9, 'o', 'Model 1')
td.add_contours([0.5, 0.8, 0.9])

# Finalize and save
td.finish_plot()
td.fig.savefig('taylor_diagram.png', dpi=300, bbox_inches='tight')
```

#### `set_xlabel(label, **kwargs)`

Set the x-axis label.

```python
td.set_xlabel('Standard Deviation', fontsize=12)
```

#### `set_ylabel(label, **kwargs)`

Set the y-axis label.

```python
td.set_ylabel('Standard Deviation', fontsize=12)
```

#### `set_title(title, **kwargs)`

Set the plot title.

```python
td.set_title('Model Performance Comparison', fontsize=14, pad=20)
```

### Properties

#### `figure`

Access the underlying matplotlib Figure object.

```python
@property
def figure(self):
    """Get the matplotlib Figure object."""
    return self._figure
```

#### `axes`

Access the matplotlib Axes object.

```python
@property
def axes(self):
    """Get the matplotlib Axes object."""
    return self._axes
```

## Usage Examples

### Basic Taylor Diagram

```python
import numpy as np
from monet_plots import taylordiagram

# Create sample data
obs_std = 1.2
model_data = [
    (1.1, 0.95, 'Model A'),
    (1.3, 0.88, 'Model B'),
    (0.9, 0.92, 'Model C'),
    (1.0, 0.97, 'Model D')
]

# Create Taylor diagram
td = taylordiagram.TaylorDiagram(
    obsstd=obs_std,
    scale=1.8,
    label='Observations'
)

# Add model samples
for stddev, corrcoef, label in model_data:
    td.add_sample(
        stddev=stddev,
        corrcoef=corrcoef,
        marker='o',
        label=label
    )

# Add contours and finalize
td.add_contours(levels=[0.5, 0.8, 0.9, 0.95])
td.finish_plot()

# Save the plot
td.save('basic_taylor_diagram.png')
```

### Advanced Taylor Diagram with Custom Styling

```python
import numpy as np
from monet_plots import taylordiagram

# Create Taylor diagram with custom styling
td = taylordiagram.TaylorDiagram(
    obsstd=2.5,
    scale=2.2,
    label='Reference Observations'
)

# Add multiple model categories
# High-performance models
high_perf_models = [
    (2.4, 0.97, 'Advanced Model', 'red', 'o'),
    (2.6, 0.95, 'Optimized Model', 'darkred', 's')
]

# Medium-performance models
med_perf_models = [
    (2.8, 0.88, 'Standard Model', 'blue', '^'),
    (2.3, 0.85, 'Baseline Model', 'lightblue', 'v')
]

# Poor-performance models
poor_perf_models = [
    (3.2, 0.72, 'Experimental Model', 'orange', 'D'),
    (3.5, 0.68, 'Prototype Model', 'darkorange', '*')
]

# Add models with different colors and markers
for models, color in [(high_perf_models, 'red'),
                      (med_perf_models, 'blue'),
for models, color in [(high_perf_models, 'red'),
                      (med_perf_models, 'blue'),
                      (poor_perf_models, 'orange')]:
    for stddev, corrcoef, label, c, marker in models:
        td.add_sample(
            stddev=stddev,
            corrcoef=corrcoef,
            marker=marker,
            label=label,
            color=c,
            markersize=8,
            zorder=10
        )

# Add custom contours
custom_levels = [0.5, 0.7, 0.8, 0.9, 0.95, 0.98]
td.add_contours(
    levels=custom_levels,
    colors='gray',
    linestyles=['-', '--', ':', '-.', (0, (3, 1, 1, 1)), (0, (5, 5))],
    linewidths=1,
    alpha=0.6
)

# Customize plot appearance
td.set_title('Comprehensive Model Performance Analysis', fontsize=16, pad=25)
td.set_xlabel('Standard Deviation', fontsize=12)
td.set_ylabel('Standard Deviation', fontsize=12)

# Add legend and finalize
td.finish_plot()

# Save high-resolution plot
td.fig.savefig('advanced_taylor_diagram.png', dpi=600, bbox_inches='tight')
```

### Integration with MONET Plots

```python
from monet_plots import TaylorDiagramPlot
import pandas as pd

# Load model evaluation data
data = pd.DataFrame({
    'model': ['Model A', 'Model B', 'Model C', 'Model D'],
    'stddev': [1.1, 1.3, 0.9, 1.0],
    'corrcoef': [0.95, 0.88, 0.92, 0.97],
    'obs_std': 1.2  # Reference standard deviation
})

# Create Taylor diagram using MONET Plots
plot = TaylorDiagramPlot(
    obsstd=data['obs_std'].iloc[0],
    scale=1.5,
    label='Observations'
)

# Add all models
for _, row in data.iterrows():
    plot.add_sample(
        df=pd.DataFrame({'obs': [1], 'model': [1]}),  # Dummy data for corrcoef
        col1='obs',
        col2='model',
        marker='o',
        label=row['model']
    )

# Update standard deviations manually
plot.dia.samples[-1].set_stddev(row['stddev'])
plot.dia.samples[-1].set_corrcoef(row['corrcoef'])

# Add contours and finish
plot.add_contours([0.5, 0.8, 0.9, 0.95])
plot.finish_plot()

plot.save('monet_taylor_diagram.png')
```

## Advanced Features

### Custom Contour Styling

```python
# Create gradient contour colors
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Create custom colormap
colors = ['#ffffcc', '#ffeda0', '#fed976', '#feb24c', '#fd8d3c', '#fc4e2a', '#e31a1c', '#bd0026', '#800026']
n_bins = 100
cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)

# Add styled contours
td.add_contours(
    levels=[0.5, 0.7, 0.8, 0.9, 0.95, 0.98],
    cmap=cmap,
    linewidths=2,
    alpha=0.8
)
```

### Interactive Taylor Diagrams

```python
# Enable interactive features
import matplotlib.pyplot as plt
plt.ion()  # Turn on interactive mode

td = taylordiagram.TaylorDiagram(obsstd=1.0)
td.add_sample(0.8, 0.9, 'o', 'Interactive Model')

# Add click interaction
def on_click(event):
    if event.inaxes == td.axes:
        # Add model at clicked position
        x, y = event.xdata, event.ydata
        r = np.sqrt(x**2 + y**2)
        if r > 0:
            corrcoef = y / r
            td.add_sample(x, corrcoef, 'x', f'Clicked Model {len(td.dia.samples)}')
            td.fig.canvas.draw()

td.fig.canvas.mpl_connect('button_press_event', on_click)
```

## Data Requirements

### Input Data Format

Taylor diagrams require two key metrics for each model:

1. **Standard Deviation**: The RMS difference between model and observations
2. **Correlation Coefficient**: The temporal correlation between model and observations

### Calculation Example

```python
import numpy as np
from scipy import stats

def calculate_taylor_metrics(obs, model):
    """Calculate standard deviation and correlation for Taylor diagram.


    Args:
        obs (array-like): Observation data
        model (array-like): Model data


    Returns:
        tuple: (stddev, corrcoef)
    """
    # Calculate standard deviations
    obs_std = np.std(obs, ddof=1)
    model_std = np.std(model, ddof=1)


    # Calculate correlation coefficient
    corrcoef, _ = stats.pearsonr(obs, model)


    return model_std, corrcoef

# Example usage
obs_data = np.random.normal(0, 1, 1000)
model_data = np.random.normal(0.1, 1.1, 1000)

stddev, corrcoef = calculate_taylor_metrics(obs_data, model_data)
print(f"Standard Deviation: {stddev:.3f}")
print(f"Correlation Coefficient: {corrcoef:.3f}")
```

## Best Practices

### Model Selection

1. **Include reference**: Always include observations as the reference point
2. **Compare like-for-like**: Ensure models are evaluated on the same dataset
3. **Statistical significance**: Only include statistically significant correlations

### Visual Design

1. **Color coding**: Use colors to indicate model categories or performance levels
2. **Marker variety**: Use different markers for different model types
3. **Clear labeling**: Ensure all models are clearly labeled
4. **Contour levels**: Choose appropriate contour levels based on your data range

### Interpretation

1. **Center point**: Models closer to the observation point are better
2. **Distance from center**: Radial distance indicates overall skill
3. **Correlation**: Higher correlation values are better
4. **Standard deviation**: Values close to observation standard deviation are better

## Common Use Cases

### Climate Model Evaluation

```python
# Compare multiple climate models
climate_models = {
    'GCM Model A': (1.05, 0.96),
    'GCM Model B': (1.15, 0.92),
    'GCM Model C': (0.95, 0.94),
    'Regional Model': (1.25, 0.88),
    'Statistical Model': (0.85, 0.91)
}

td = taylordiagram.TaylorDiagram(obsstd=1.0, label='Reanalysis')
for name, (std, corr) in climate_models.items():
    td.add_sample(std, corr, 'o', name)

td.add_contours([0.5, 0.8, 0.9, 0.95])
td.finish_plot()
```

### Weather Forecast Verification

```python
# Compare different forecast lead times
forecast_models = {
    '24h Forecast': (0.8, 0.98),
    '48h Forecast': (1.0, 0.94),
    '72h Forecast': (1.2, 0.88),
    '96h Forecast': (1.4, 0.82),
    '120h Forecast': (1.6, 0.75)
}

td = taylordiagram.TaylorDiagram(obsstd=1.0, label='Observations')
for name, (std, corr) in forecast_models.items():
    td.add_sample(std, corr, 's', name)

td.add_contours([0.5, 0.8, 0.9, 0.95])
td.finish_plot()
```

---

**Related Resources**:

- [Plot Types API](../plots/index.md) - Other plot implementations
- [Examples](../examples/index.md) - Practical usage examples
- [Base API](./base.md) - Core plotting functionality
