"""
Facet Grid Spatial Simplified
=============================

This example demonstrates Facet Grid Spatial Simplified.
"""

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from monet_plots.plots.facet_grid import FacetGridPlot
from monet_plots.plots.spatial_imshow import SpatialImshowPlot
from typing import List


# --- 1. Helper function to create the meta DataFrame ---
def create_facet_grid_data(
    data_arrays: List[xr.DataArray],
    facet_attrs: List[str],
    data_var_name: str = "data_array",
) -> pd.DataFrame:
    """
    Creates a pandas DataFrame suitable for FacetGridPlot from a list of xarray.DataArrays.
    Facet information is extracted from DataArray attributes.

    Args:
        data_arrays: A list of xarray.DataArray objects, each with facet attributes.
        facet_attrs: A list of attribute names (strings) to use as facet dimensions.
        data_var_name: The name of the column that will hold the DataArray objects.

    Returns:
        A pandas DataFrame where each row represents a facet, containing facet attributes
        and the corresponding DataArray object.
    """
    records = []
    for da in data_arrays:
        record = {attr: da.attrs.get(attr) for attr in facet_attrs}
        record[data_var_name] = da
        records.append(record)
    return pd.DataFrame(records)


# --- 2. Simulate heterogeneous data with attributes ---
# Model A data (coarser resolution)
lats_A = np.arange(30, 35, 1.0)
lons_A = np.arange(-100, -95, 1.0)
time_A = pd.to_datetime(["2023-01-01T00:00", "2023-01-01T06:00"])
da_A_t0 = xr.DataArray(
    np.random.rand(len(lats_A), len(lons_A)) * 50 + 273.15,  # K
    coords={"lat": lats_A, "lon": lons_A},
    dims=["lat", "lon"],
    name="temperature_K",
)
da_A_t0.attrs["model"] = "Model A"
da_A_t0.attrs["time_label"] = time_A[0].strftime("%Y-%m-%d %H:%M")

da_A_t1 = xr.DataArray(
    np.random.rand(len(lats_A), len(lons_A)) * 50 + 273.15,  # K
    coords={"lat": lats_A, "lon": lons_A},
    dims=["lat", "lon"],
    name="temperature_K",
)
da_A_t1.attrs["model"] = "Model A"
da_A_t1.attrs["time_label"] = time_A[1].strftime("%Y-%m-%d %H:%M")


# Model B data (finer resolution)
lats_B = np.arange(30, 35, 0.5)
lons_B = np.arange(-100, -95, 0.5)
time_B = pd.to_datetime(["2023-01-01T03:00", "2023-01-01T09:00"])
da_B_t0 = xr.DataArray(
    np.random.rand(len(lats_B), len(lons_B)) * 30 + 280.15,  # K
    coords={"lat": lats_B, "lon": lons_B},
    dims=["lat", "lon"],
    name="temperature_K",
)
da_B_t0.attrs["model"] = "Model B"
da_B_t0.attrs["time_label"] = time_B[0].strftime("%Y-%m-%d %H:%M")

da_B_t1 = xr.DataArray(
    np.random.rand(len(lats_B), len(lons_B)) * 30 + 280.15,  # K
    coords={"lat": lats_B, "lon": lons_B},
    dims=["lat", "lon"],
    name="temperature_K",
)
da_B_t1.attrs["model"] = "Model B"
da_B_t1.attrs["time_label"] = time_B[1].strftime("%Y-%m-%d %H:%M")

# Collect all DataArrays into a list
all_data_arrays = [da_A_t0, da_A_t1, da_B_t0, da_B_t1]

# --- 3. Use the helper function to create the meta DataFrame ---
meta_df = create_facet_grid_data(all_data_arrays, facet_attrs=["model", "time_label"])


# --- 4. Define a custom plotting function ---
def plot_heterogeneous_spatial(data, **kwargs):
    # 'data' here is a DataFrame subset for a single facet.
    # It will have only one row in this setup.

    # Extract the actual xarray.DataArray from the 'data_array' column
    da_to_plot = data["data_array"].iloc[0]

    # Create and plot using SpatialImshowPlot
    plotter = SpatialImshowPlot(da_to_plot, **kwargs)
    plotter.plot()


# --- 5. Create the FacetGridPlot using the Meta DataFrame ---
grid = FacetGridPlot(
    meta_df,
    row="time_label",
    col="model",
    height=4,
    aspect=1.2,
    cbar_label="Temperature (K)",
)

# --- 6. Map the custom plotting function to the grid ---
grid.map_dataframe(plot_heterogeneous_spatial, cmap="plasma", add_colorbar=True)

# --- 7. Set titles and adjust layout ---
grid.set_titles(col_template="{col_name}", row_template="{row_name}")
plt.tight_layout()

# --- 8. Save the figure ---
grid.save("facet_grid_heterogeneous_spatial_plot_simplified.png", dpi=300)

print("Generated 'facet_grid_heterogeneous_spatial_plot_simplified.png'")
