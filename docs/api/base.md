# Base Plot Module

The `monet_plots.base` module provides the foundational classes and functionality for all plot types in MONET Plots.

## BasePlot Class

`BasePlot` is the abstract base class that all plot types inherit from. It provides common functionality for plot creation, customization, and management.

### Class Signature

```python
class BasePlot:
    """Abstract base class for all plot types in MONET Plots."""


    def __init__(self, figsize=(8, 6), dpi=100, **kwargs):
        """Initialize a BasePlot instance.


        Args:
            figsize (tuple): Figure size (width, height) in inches
            dpi (int): Dots per inch for the figure
            **kwargs: Additional keyword arguments
        """
        pass
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `figsize` | `tuple` | `(8, 6)` | Figure size in inches (width, height) |
| `dpi` | `int` | `100` | Resolution of the figure in dots per inch |
| `**kwargs` | `dict` | `{}` | Additional matplotlib figure parameters |

### Public Methods

#### `plot(data, **kwargs)`

Main plotting method that must be implemented by subclasses.

```python
def plot(self, data, **kwargs):
    """Main plotting method.


    Args:
        data: Input data to plot
        **kwargs: Additional plotting parameters


    Returns:
        None
    """
    raise NotImplementedError("Subclasses must implement plot method")
```

#### `save(filename, dpi=None, **kwargs)`

Save the current plot to a file.

```python
def save(self, filename, dpi=None, **kwargs):
    """Save the plot to a file.


    Args:
        filename (str): Output filename
        dpi (int, optional): DPI for saved image. Defaults to Figure DPI
        **kwargs: Additional savefig parameters


    Returns:
        None
    """
    pass
```

**Example:**
```python
plot = SpatialPlot()
plot.plot(data)
plot.save('output.png', dpi=300)  # High resolution output
```

#### `close()`

Close the plot and free memory.

```python
def close(self):
    """Close the plot and free resources.


    Returns:
        None
    """
    pass
```

**Example:**
```python
plot = SpatialPlot()
plot.plot(data)
plot.save('output.png')
plot.close()  # Free memory
```

#### `title(text, fontsize=14, **kwargs)`

Set the plot title.

```python
def title(self, text, fontsize=14, **kwargs):
    """Set the plot title.


    Args:
        text (str): Title text
        fontsize (int, optional): Font size. Defaults to 14
        **kwargs: Additional title parameters


    Returns:
        Self for method chaining
    """
    pass
```

**Example:**
```python
plot = SpatialPlot()
plot.plot(data)
plot.title("My Plot Title", fontsize=16, pad=20)
```

#### `xlabel(text, fontsize=12, **kwargs)`

Set the x-axis label.

```python
def xlabel(self, text, fontsize=12, **kwargs):
    """Set the x-axis label.


    Args:
        text (str): X-axis label text
        fontsize (int, optional): Font size. Defaults to 12
        **kwargs: Additional label parameters


    Returns:
        Self for method chaining
    """
    pass
```

**Example:**
```python
plot = TimeSeriesPlot()
plot.plot(df, x='time', y='value')
plot.xlabel("Date", fontsize=12)
```

#### `ylabel(text, fontsize=12, **kwargs)`

Set the y-axis label.

```python
def ylabel(self, text, fontsize=12, **kwargs):
    """Set the y-axis label.


    Args:
        text (str): Y-axis label text
        fontsize (int, optional): Font size. Defaults to 12
        **kwargs: Additional label parameters


    Returns:
        Self for method chaining
    """
    pass
```

**Example:**
```python
plot = TimeSeriesPlot()
plot.plot(df, x='time', y='value')
plot.ylabel("Temperature (°C)")
```

#### `legend(*args, **kwargs)`

Add a legend to the plot.

```python
def legend(self, *args, **kwargs):
    """Add a legend to the plot.


    Args:
        *args: Legend labels
        **kwargs: Additional legend parameters


    Returns:
        Self for method chaining
    """
    pass
```

**Example:**
```python
plot = ScatterPlot()
plot.plot(data1, label='Model 1')
plot.plot(data2, label='Model 2')
plot.legend(loc='upper right', fontsize=10)
```

#### `grid(show=True, **kwargs)`

Toggle plot grid.

```python
def grid(self, show=True, **kwargs):
    """Toggle plot grid.


    Args:
        show (bool): Whether to show grid. Defaults to True
        **kwargs: Additional grid parameters


    Returns:
        Self for method chaining
    """
    pass
```

**Example:**
```python
plot = ScatterPlot()
plot.plot(data)
plot.grid(show=True, linestyle='--', alpha=0.5)
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

Access the matplotlib Axes object(s).

```python
@property
def axes(self):
    """Get the matplotlib Axes object(s)."""
    return self._axes
```

### Method Chaining

All setter methods return `self`, allowing method chaining:

```python
plot = SpatialPlot()
plot.plot(data)\
   .title("My Plot")\
   .xlabel("X-axis")\
   .ylabel("Y-axis")\
   .legend()\
   .save("output.png")
```

### Subclass Implementation

When creating custom plot classes, inherit from `BasePlot` and implement the `plot()` method:

```python
from monet_plots.base import BasePlot
import matplotlib.pyplot as plt

class CustomPlot(BasePlot):
    def plot(self, data, **kwargs):
        """Implement custom plotting logic."""
        # Your plotting code here
        plt.plot(data, **kwargs)
        return self
```

### Error Handling

The base class includes automatic error handling for common issues:

- **Invalid filenames**: Raises `ValueError` for invalid file paths
- **Missing dependencies**: Provides helpful error messages
- **Memory management**: Automatic resource cleanup when possible

### Debug Mode

Enable debug output for troubleshooting:

```python
plot = SpatialPlot(debug=True)
plot.plot(data)
```

### Performance Considerations

- Always call [`close()`](#close) when done with plots to free memory
- Reuse plot objects when possible instead of creating new ones
- Use appropriate DPI values for your output needs

---

**Related Resources**:

- [Plot Types API](../plots/index.md) - Specific plot type implementations
- [Style Configuration](../style) - Customizing plot appearance
- [Examples](../examples/index.md) - Practical usage examples
