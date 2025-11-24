# Examples and Tutorials

Welcome to the MONET Plots examples section! This section provides comprehensive tutorials and practical examples to help you master the library for various scientific visualization tasks.

## Overview

The examples are organized by difficulty and application area, starting from basic usage and progressing to advanced multi-plot workflows.

| Example Category | Difficulty | Focus Area |
|------------------|------------|------------|
| [Getting Started](./getting-started) | Beginner | Basic plotting concepts |
| [Core Plot Types](./core-plots) | Beginner | Individual plot types |
| [Brier Score Decomposition](./plots/brier_decomposition.md) | Intermediate | Forecast verification |
| [Categorical Plot](./plots/categorical_plot.md) | Intermediate | Data distribution |
| [Conditional Bias Plot](./plots/conditional_bias_plot.md) | Intermediate | Forecast bias analysis |
| [Spread-Skill Plot](./plots/spread_skill_plot.md) | Intermediate | Ensemble reliability |
| [Advanced Workflows](./advanced-workflows) | Intermediate | Multi-plot analysis |
| [Scientific Applications](./scientific-applications) | Advanced | Domain-specific use cases |
| [Customization](./customization) | Intermediate | Advanced styling and customization |

## Learning Path

### 1. Beginner Path
If you're new to MONET Plots, follow this path:

1. **[Basic Plotting](./getting-started/basic-plotting)** - Learn the fundamentals
2. **[Data Preparation](./getting-started/data-preparation)** - Understand data formats
3. **[Plot Customization](./getting-started/plot-customization)** - Basic styling

### 2. Intermediate Path
For users with some experience:

1. **[Core Plot Types](./core-plots)** - Master individual plot types
2. **[Brier Score Decomposition](./plots/brier_decomposition.md)** - Analyze forecast skill
3. **[Categorical Plot](./plots/categorical_plot.md)** - Visualize data distribution
4. **[Conditional Bias Plot](./plots/conditional_bias_plot.md)** - Analyze forecast bias
5. **[Spread-Skill Plot](./plots/spread_skill_plot.md)** - Evaluate ensemble reliability
2. **[Multi-Plot Layouts](./advanced-workflows/multi-plot-layouts)** - Combine plots
3. **[Statistical Analysis](./advanced-workflows/statistical-analysis)** - Add statistics

### 3. Advanced Path
For experienced users:

1. **[Scientific Applications](./scientific-applications)** - Domain-specific examples
2. **[Performance Optimization](./advanced-workflows/performance)** - Large datasets
3. **[Custom Extensions](./customization/custom-classes)** - Create your own plots

## Example Structure

Each example follows a consistent structure:

### Setup Section
- **Objective**: What you'll learn
- **Prerequisites**: What you need to know
- **Data Requirements**: What kind of data is needed

### Implementation
- **Step-by-step code** with explanations
- **Expected output** with visual descriptions
- **Key concepts** highlighted

### Common Patterns
- **Best practices** for this type of plot
- **Troubleshooting tips** for common issues
- **Variations** to try

### Next Steps
- **Related examples** to explore
- **Advanced topics** to learn
- **Real-world applications**

## Running the Examples

### Interactive Jupyter Notebooks
All examples are available as Jupyter notebooks with executable code:

```bash
# Clone the repository
git clone https://github.com/your-repo/monet-plots.git
cd monet-plots

# Start Jupyter
jupyter notebook docs/examples/

# Or start Jupyter Lab
jupyter lab docs/examples/
```

### Scripts for Reproducibility
Examples are also provided as standalone Python scripts:

```bash
# Run individual examples
python docs/examples/getting-started/basic-plotting.py

# Run all examples
python docs/examples/run_all_examples.py
```

## Data Used in Examples

### Synthetic Data
Most examples use synthetic data for reproducibility:

```python
import numpy as np
import pandas as pd

# Common data generation patterns
def create_timeseries_data(n_points=100):
    """Create synthetic time series data."""
    dates = pd.date_range('2023-01-01', periods=n_points, freq='D')
    values = np.cumsum(np.random.normal(0, 1, n_points)) + 50
    return pd.DataFrame({'date': dates, 'value': values})

def create_spatial_data(n_lat=20, n_lon=30):
    """Create synthetic spatial data."""
    lat = np.linspace(30, 50, n_lat)
    lon = np.linspace(-120, -70, n_lon)
    data = np.random.random((n_lat, n_lon)) * 100
    return data, lat, lon
```

### Real Data Examples
Some examples use real meteorological datasets:

```python
import xarray as xr

# Load sample climate data
def load_climate_data():
    """Load sample climate data."""
    # This would load actual NetCDF files in real examples
    # For now, we create synthetic data
    pass
```

## Contributing Examples

We welcome community contributions! To add a new example:

1. **Choose a category** from the existing structure
2. **Follow the template** for consistent formatting
3. **Test your code** thoroughly
4. **Add screenshots** of the output
5. **Submit a pull request** with detailed description

### Example Template
```python
"""
Example: [Your Example Name]

Description:
[Brief description of what this example demonstrates]

Objective:
[What the reader will learn]

Prerequisites:
- [Required knowledge]
- [Other examples to complete first]

Steps:
1. [First step with explanation]
2. [Second step with explanation]

Expected Output:
[Description of the plot/visualization]

Key Concepts:
- [Concept 1]
- [Concept 2]

Next Steps:
- [Related examples]
- [Advanced topics to explore]
"""

# Your code here
```

## Troubleshooting

### Common Issues
- **Import errors**: Ensure all dependencies are installed
- **Missing data**: Check data formats and file paths
- **Plot display issues**: Verify matplotlib backend settings
- **Memory errors**: Consider downsampling large datasets

### Getting Help
- **Documentation**: Check the main [API Reference](../api)
- **Issues**: Report problems on [GitHub Issues](https://github.com/your-repo/monet-plots/issues)
- **Discussions**: Join [GitHub Discussions](https://github.com/your-repo/monet-plots/discussions)

## Featured Examples

### Climate Science
- [Temperature Trend Analysis](./scientific-applications/climate-trends)
- [Precipitation Patterns](./scientific-applications/precipitation-analysis)
- [Wind Field Analysis](./scientific-applications/wind-fields)

### Model Evaluation
- [Taylor Diagram Comparison](./advanced-workflows/taylor-diagrams)
- [Bias Assessment](./advanced-workflows/bias-analysis)
- [Skill Score Calculation](./advanced-workflows/skill-scores)

### Data Visualization
- [Multi-Panel Layouts](./advanced-workflows/multi-plot-layouts)
- [Interactive Plots](./advanced-workflows/interactive-plots)
- [Animation Examples](./advanced-workflows/animated-plots)

## Quick Start

### Your First Plot
```python
import numpy as np
from monet_plots import SpatialPlot

# Create a simple spatial plot
plot = SpatialPlot(figsize=(10, 8))

# Generate data
data = np.random.random((20, 30)) * 100

# Plot and save
plot.plot(data, title="My First Plot")
plot.save("first_plot.png")
plot.close()

print("Plot saved successfully!")
```

### Next Steps
1. Explore [Basic Plotting](./getting-started/basic-plotting) for more fundamentals
2. Try [Customization](./customization) to make plots your own
3. Check [Advanced Workflows](./advanced-workflows) for complex analyses

---

**Navigation**:

- [Getting Started Guide](../getting-started) - Installation and setup
- [API Reference](../api) - Complete API documentation
- [Plot Types](../plots) - Individual plot type documentation
- [Configuration Guide](../configuration) - Styling and customization
- [Performance Guide](../performance) - Optimization techniques