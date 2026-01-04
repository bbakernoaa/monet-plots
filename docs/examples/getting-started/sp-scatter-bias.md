# Spatial Scatter Bias Plots

Spatial scatter bias plots are a specialized visualization used to display the geographical distribution of differences (bias) between two datasets. They are particularly effective for identifying regions where a model or forecast systematically over- or under-predicts compared to observations or a reference. The size of the plotted points is scaled by the magnitude of the difference, making larger discrepancies visually prominent, while the color indicates the sign of the bias.

## Prerequisites

-   Basic Python knowledge
-   Understanding of numpy and pandas
-   MONET Plots installed (`pip install monet_plots`)
-   Cartopy installed (`pip install cartopy`)
-   Matplotlib installed (`pip install matplotlib`)

## Plotting Workflow

To create a spatial scatter bias plot with `monet_plots`:

1.  **Prepare Data**: Your data should be in a `pandas.DataFrame` containing `latitude`, `longitude`, and two numerical columns representing the reference and comparison values.
2.  **Initialize `SpScatterBiasPlot`**: Create an instance of the `SpScatterBiasPlot` class, passing your DataFrame and the names of the reference (`col1`) and comparison (`col2`) columns.
3.  **Call `plot` method**: Generate the spatial scatter plot.
4.  **Customize (Optional)**: Adjust map projections, colors, point sizes, or other visual enhancements.
5.  **Display/Save**: Show the plot using `plt.show()` or save it to a file.

## Example 1: Basic Spatial Scatter Bias Plot

Let's create a basic spatial scatter bias plot using synthetic data to visualize differences across a geographical region.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from monet_plots.plots.sp_scatter_bias import SpScatterBiasPlot

# 1. Prepare sample data
np.random.seed(42) # for reproducibility
n_points = 500

# Simulate random latitude and longitude points
lat = np.random.uniform(20, 50, n_points)
lon = np.random.uniform(-120, -70, n_points)

# Simulate reference and comparison values
reference_values = 10 + 5 * np.random.rand(n_points)
# Introduce a spatial bias: higher values in the west, lower in the east
comparison_values = reference_values + (lon / 100 + np.random.normal(0, 0.5, n_points))

df = pd.DataFrame({
    'latitude': lat,
    'longitude': lon,
    'reference_value': reference_values,
    'comparison_value': comparison_values
})

# 2. Initialize and create the plot
plot = SpScatterBiasPlot(
    df=df,
    col1='reference_value',
    col2='comparison_value',
    figsize=(10, 8),
    map_kwargs={'projection': ccrs.PlateCarree(), 'extent': [-125, -65, 15, 55]} # Focus on North America
)
plot.plot(
    cmap='RdBu_r', # Red-Blue colormap, reversed for positive=red, negative=blue
    edgecolor='black',
    linewidth=0.5,
    alpha=0.8
)

# 3. Add titles and labels
plot.ax.set_title("Spatial Bias Plot: Comparison vs. Reference")
# Map elements (coastlines, borders) are added by default by the SpatialPlot class

plt.tight_layout()
plt.show()
```

### Expected Output

A geographical map of North America will be displayed. Scatter points will be plotted at various latitude and longitude locations. The color of each point will indicate the bias (comparison - reference value), with a diverging colormap (e.g., red for positive bias, blue for negative bias). The size of each point will be proportional to the absolute magnitude of the bias, making areas with larger differences stand out. A colorbar will be present to interpret the bias values.

### Key Concepts

-   **`SpScatterBiasPlot`**: The class used to generate spatial scatter bias plots.
-   **`col1`**: The name of the column containing the reference values.
-   **`col2`**: The name of the column containing the comparison values.
-   **`map_kwargs`**: A dictionary to pass arguments to `monet_plots.plots.spatial.SpatialPlot.from_projection`, allowing customization of the map projection, features, and extent.
-   **Point Size and Color**: Point size visually emphasizes the magnitude of the bias, while color indicates its sign (over- or under-prediction).

## Example 2: Customizing Map and Plot Appearance

You can further customize the map projection, add specific map features, and adjust the appearance of the scatter points and colorbar.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from monet_plots.plots.sp_scatter_bias import SpScatterBiasPlot

# 1. Prepare sample data (same as Example 1)
np.random.seed(42) # for reproducibility
n_points = 500
lat = np.random.uniform(20, 50, n_points)
lon = np.random.uniform(-120, -70, n_points)
reference_values = 10 + 5 * np.random.rand(n_points)
comparison_values = reference_values + (lon / 100 + np.random.normal(0, 0.5, n_points))
df = pd.DataFrame({
    'latitude': lat,
    'longitude': lon,
    'reference_value': reference_values,
    'comparison_value': comparison_values
})

# 2. Initialize and create the plot with custom map features and plot styles
plot = SpScatterBiasPlot(
    df=df,
    col1='reference_value',
    col2='comparison_value',
    figsize=(12, 10),
    # Custom map projection and features
    map_kwargs={
        'projection': ccrs.AlbersEqualArea(central_longitude=-98, central_latitude=38),
        'extent': [-125, -65, 15, 55],
        'features': [cfeature.STATES, cfeature.BORDERS, cfeature.COASTLINE, cfeature.LAKES]
    },
    val_min=-2, # Set explicit min/max for color scaling
    val_max=2
)
plot.plot(
    cmap='coolwarm', # Another diverging colormap
    edgecolor='gray',
    linewidth=0.2,
    alpha=0.9
)

# 3. Add titles and labels
plot.ax.set_title("Customized Spatial Bias Plot (Albers Equal Area)", fontsize=16)
# Map elements (coastlines, borders) are added by default by the SpatialPlot class

plt.tight_layout()
plt.show()
```

### Expected Output

A spatial scatter bias plot similar to Example 1, but with a different map projection (Albers Equal Area), potentially more detailed map features (states, lakes), and a customized colormap ('coolwarm'). The scatter points might have different edge colors and linewidths, and the colorbar could be horizontal with a custom label. The `val_min` and `val_max` parameters will ensure the color scale is consistent, regardless of the data's actual min/max.

### Key Concepts

-   **`map_kwargs['projection']`**: Allows specifying any `cartopy.crs` projection for the map.
-   **`map_kwargs['features']`**: A list of `cartopy.feature` objects to add to the map (e.g., `cfeature.STATES`, `cfeature.LAKES`).
-   **`val_min` and `val_max`**: Explicitly set the minimum and maximum values for the color scale, which can be useful for comparing multiple plots with consistent color ranges.
