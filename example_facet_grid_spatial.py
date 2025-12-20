import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from monet_plots.plots.facet_grid import FacetGridPlot
from monet_plots.plots.spatial_imshow import SpatialImshowPlot

# 1. Create some dummy spatial data
# Simulate data for two models ('model_A', 'model_B') and two times ('2023-01-01', '2023-01-02')
lats = np.arange(30, 35, 0.5)
lons = np.arange(-100, -95, 0.5)
times = pd.to_datetime(['2023-01-01', '2023-01-02'])
models = ['model_A', 'model_B']

# Create a DataArray with dimensions (time, model, lat, lon)
data = xr.DataArray(
    np.random.rand(len(times), len(models), len(lats), len(lons)) * 100,
    coords={
        'time': times,
        'model': models,
        'lat': lats,
        'lon': lons
    },
    dims=['time', 'model', 'lat', 'lon'],
    name='temperature'
)

# Add some variation for demonstration
data.loc[{'model': 'model_A', 'time': '2023-01-01'}] += 10
data.loc[{'model': 'model_B', 'time': '2023-01-02'}] -= 5

# Convert to Dataset if you have multiple variables, or keep as DataArray
ds = data.to_dataset()

# 2. Define a plotting function for a single spatial plot
# This function will be mapped to each facet.
# It needs to accept a DataFrame (or similar) and the FacetGrid will pass the subsetted data.


def plot_spatial_imshow(data, **kwargs):
    # The data passed to this function by map_dataframe will be a DataFrame
    # We need to convert it back to an xarray DataArray for SpatialImshowPlot
    # Assuming 'lat' and 'lon' are present in the DataFrame

    # Reconstruct DataArray from DataFrame for plotting
    # This step might need adjustment based on your actual data structure
    # and how map_dataframe passes it. For simplicity, we'll assume
    # the DataFrame has 'lat', 'lon', and the variable 'temperature'.

    # Get the variable name from the original dataset
    var_name = ds.data_vars[0] if isinstance(ds, xr.Dataset) else ds.name

    # Create a temporary DataArray for plotting
    # Ensure 'lat' and 'lon' are correctly identified as coordinates
    temp_da = data.set_index(['lat', 'lon']).to_xarray()[var_name]

    # Create and plot using SpatialImshowPlot
    plotter = SpatialImshowPlot(temp_da, **kwargs)
    plotter.plot()
    # The figure and axes are managed by FacetGrid, so we don't call plotter.show() or plotter.save() here.
    # We just ensure the plot is drawn on the current active axes, which FacetGrid handles.


# 3. Create the FacetGridPlot
# We want 'time' as rows and 'model' as columns
grid = FacetGridPlot(
    ds,
    row='time',
    col='model',
    height=4,
    aspect=1.2,
    cbar_label='Temperature'  # This will be passed to the spatial plot
)

# 4. Map the spatial plotting function to the grid
# Pass the variable name to plot_spatial_imshow
grid.map_dataframe(plot_spatial_imshow, 'temperature',
                   cmap='viridis', add_colorbar=True)

# 5. Set titles and adjust layout
grid.set_titles(col_template="{col_name}", row_template="{row_name}")
plt.tight_layout()

# 6. Save the figure
grid.save("facet_grid_spatial_plot.png", dpi=300)

print("Generated 'facet_grid_spatial_plot.png'")
