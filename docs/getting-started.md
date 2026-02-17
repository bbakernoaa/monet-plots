# Getting Started Guide

Welcome to MONET Plots! This comprehensive guide will help you get up and running quickly with our scientific plotting library, from installation to creating your first professional plots.

## Installation

### System Requirements

- **Python**: 3.7 or higher (recommended 3.8+)
- **pip**: Python package manager (comes with Python)
- **Optional**: conda (for managing complex dependencies)
- **Memory**: Minimum 512MB RAM (1GB+ recommended for large datasets)
- **Disk**: 50MB free space for installation

### Prerequisites

Before installing MONET Plots, ensure you have the following:

```bash
# Check Python version
python --version  # Should be 3.7 or higher

# Check pip is installed and working
pip --version

# Update pip to latest version
python -m pip install --upgrade pip
```

### Step-by-Step Installation

#### 1. Install MONET Plots

```bash
# Install the latest stable release from PyPI
pip install monet_plots

# For the latest development version
pip install git+https://github.com/your-repo/monet-plots.git

# Install specific version (if needed)
pip install monet_plots==1.0.0
```

#### 2. Install Optional Dependencies

MONET Plots has a modular design with optional dependencies for enhanced functionality. Install these based on your needs:

**Core Dependencies (Recommended for most users):**
```bash
# Essential data processing and visualization
pip install pandas seaborn matplotlib

# Statistical computing
pip install numpy scipy
```

**Geospatial and Climate Data:**
```bash
# For geographical maps and projections
pip install cartopy

# For NetCDF file handling
pip install xarray netcdf4

# For advanced geospatial operations
pip install shapely pyproj
```

**Advanced Statistical Features:**
```bash
# For machine learning integration
pip install scikit-learn

# For advanced statistical analysis
pip install statsmodels
```

**Development and Testing:**
```bash
# For development
pip install pytest black isort mypy

# Documentation generation
pip install mkdocs mkdocs-material
```

#### 3. Verify Installation

Test your installation with these verification steps:

```python
# Basic import test
import monet_plots
print(f"MONET Plots version: {monet_plots.__version__}")

# Test all major components
try:
    from monet_plots import (
        SpatialPlot, TimeSeriesPlot, ScatterPlot,
        TaylorDiagramPlot, KDEPlot, WindQuiverPlot, WindBarbsPlot,
        FacetGridPlot
    )
    print("All plot classes imported successfully!")
except ImportError as e:
    print(f"Import error: {e}")

# Test optional dependencies
try:
    import cartopy
    print("Cartopy available for geospatial plotting")
except ImportError:
    print("Cartopy not available - install for geospatial features")

try:
    import xarray
    print("xarray available for NetCDF support")
except ImportError:
    print("xarray not available - install for NetCDF support")
```

You should see success messages for all components without any errors.

#### 4. Environment Setup

**Virtual Environment (Recommended):**
```bash
# Create virtual environment
python -m venv monet_plots_env

# Activate virtual environment
# On Windows
monet_plots_env\Scripts\activate
# On macOS/Linux
source monet_plots_env/bin/activate

# Install in virtual environment
pip install monet_plots
```

**Conda Environment:**
```bash
# Create conda environment
conda create -n monet_plots python=3.8
conda activate monet_plots

# Install with conda
conda install -c conda-forge monet_plots
```

#### 5. Common Installation Issues

**Permission Errors:**
```bash
# Use user install (no sudo)
pip install --user monet_plots

# Or use virtual environment
python -m venv venv
source venv/bin/activate
pip install monet_plots
```

**Network Issues:**
```bash
# Use pip with timeout
pip install --timeout=60 monet_plots

# Or use conda
conda install -c conda-forge monet_plots
```

**Missing Dependencies:**
```bash
# Install specific versions
pip install matplotlib>=3.3.0
pip install pandas>=1.0.0
pip install numpy>=1.18.0
pip install monet_plots
```

## Quick Setup

### Basic Configuration

```python
import matplotlib.pyplot as plt
from monet_plots import style

# Apply the default Wiley-compliant style
plt.style.use(style.wiley_style)

# Your plotting code here
```

### Environment Setup

For consistency across projects, consider adding this to your environment setup:

```python
# setup_plotting.py
import matplotlib.pyplot as plt
from monet_plots import style

# Set up plotting style and defaults
plt.style.use(style.wiley_style)
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.format'] = 'png'
```

## Your First Plot

Let's create a simple time series plot to demonstrate the basic workflow:

```python
import pandas as pd
import numpy as np
from monet_plots import TimeSeriesPlot

# Create sample data
dates = pd.date_range('2023-01-01', periods=100, freq='D')
data = pd.DataFrame({
    'time': dates,
    'temperature': 20 + 10 * np.sin(np.arange(100) * 0.1) + np.random.normal(0, 2, 100)
})

# Create and plot
plot = TimeSeriesPlot()
plot.plot(data, x='time', y='temperature', title='Daily Temperature', ylabel='Temperature (°C)')
plot.save('temperature_plot.png')
plot.close()
```

## Understanding the Architecture

### Plot Classes

MONET Plots follows a class-based architecture where each plot type has its own class:

```
BasePlot (abstract base class)
├── SpatialPlot
├── TimeSeriesPlot
├── TimeSeriesPlot
├── ScatterPlot
├── TaylorDiagramPlot
├── KDEPlot
├── WindQuiverPlot
└── WindBarbsPlot
```

### Common Workflow

1. **Initialize**: Create a plot instance with desired parameters
2. **Plot**: Call the plot method with your data
3. **Customize**: Add labels, titles, and other customizations
4. **Save**: Export the plot to file
5. **Close**: Close the plot to free memory

```python
# General workflow pattern
plot = SpatialPlot(figsize=(10, 6))         # 1. Initialize
plot.plot(data, cmap='viridis')             # 2. Plot data
plot.title("My Plot").xlabel("X Axis")      # 3. Customize
plot.save("output.png")                    # 4. Save
plot.close()                               # 5. Close
```

## Data Requirements

### Supported Data Types

#### Pandas DataFrame
Most plots expect pandas DataFrames with specific column names:

```python
df = pd.DataFrame({
    'time': pd.date_range('2023-01-01', periods=100),
    'observed': np.random.normal(0, 1, 100),
    'modeled': np.random.normal(0.1, 1.1, 100)
})
```

#### NumPy Arrays
For some plots, raw numpy arrays are sufficient:

```python
import numpy as np
data = np.random.random((50, 100))  # 2D array for spatial plots
```

#### xarray DataArrays
For geospatial data:

```python
import xarray as xr
data = xr.DataArray(
    np.random.random((10, 20)),
    dims=['lat', 'lon'],
    coords={'lat': range(10), 'lon': range(20)}
)

# New accessor-based plotting (Recommended for xarray)
data.mplots.imshow(coastlines=True)
```

### Data Format Guidelines

- **Time data**: Use pandas datetime objects for time series
- **Spatial data**: Follow standard coordinate conventions (lat/lon, x/y)
- **Missing values**: Handle NaN values appropriately
- **Units**: Include units in column names or documentation

## Configuration Options

### Style Customization

```python
import matplotlib.pyplot as plt

# Create custom style
custom_style = {
    'font.size': 12,
    'axes.labelsize': 10,
    'axes.titlesize': 14,
    'lines.linewidth': 2,
    'figure.figsize': (10, 6)
}

plt.style.use(custom_style)
```

### Default Parameters

Set up default parameters for your project:

```python
from monet_plots import style

# Modify the default style
modified_style = style.wiley_style.copy()
modified_style['font.size'] = 11
modified_style['figure.figsize'] = (12, 8)

plt.style.use(modified_style)
```

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# If you get import errors, try reinstalling
pip uninstall monet_plots
pip install monet_plots
```

#### Missing Dependencies
```bash
# Install missing cartopy dependency
pip install cartopy
```

#### Plot Display Issues
```python
# Ensure interactive plotting is enabled
%matplotlib inline  # For Jupyter notebooks
plt.ion()           # For interactive scripts
```

### Performance Tips

1. **Close plots when done**: `plot.close()`
2. **Use appropriate data types**: Pandas DataFrames for tabular data
3. **Limit data size**: Downsample large datasets for interactive plotting
4. **Use efficient file formats**: PNG for web, TIFF for publications

## Next Steps

After completing this guide, explore:

1. **[Plot Types](./plots/index.md)**: Learn about specific plot types and their usage
2. **[API Reference](./api/index.md)**: Detailed documentation for all modules
3. **[Examples](./examples/index.md)**: Practical examples and tutorials
4. **[Configuration](./configuration/index.md)**: Advanced customization options

## Need Help?

- Check the [API Reference](./api/index.md) for detailed documentation
- Browse [Examples](./examples/index.md) for use cases similar to yours
- Visit our [GitHub Issues](https://github.com/your-repo/monet-plots/issues) for support
