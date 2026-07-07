import datetime
from unittest.mock import MagicMock

import cartopy.crs as ccrs
import matplotlib.collections as mcoll
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from monet_plots.plots.spatial_bias_scatter import SpatialBiasScatterPlot


def test_spatial_bias_scatter_plot():
    """Test the SpatialBiasScatterPlot plot method."""
    # Create a mock basemap object
    mock_map = MagicMock()
    mock_map.return_value = (np.random.rand(10), np.random.rand(10))

    # Create a sample dataframe
    df = pd.DataFrame(
        {
            "latitude": np.arange(30, 40),
            "longitude": np.arange(-100, -90),
            "CMAQ": np.random.rand(10),
            "Obs": np.random.rand(10),
            "datetime": [datetime.datetime(2020, 1, 1)] * 10,
        }
    )

    # Create a SpatialBiasScatterPlot instance
    plot = SpatialBiasScatterPlot(df, col1="Obs", col2="CMAQ")

    # Call the plot method
    cbar = plot.plot()

    # Assert that the plot objects are created
    assert cbar is not None


def test_spatial_bias_scatter_on_existing_ax():
    """Test that SpatialBiasScatterPlot can draw on a pre-existing GeoAxes."""
    df = pd.DataFrame(
        {
            "latitude": np.arange(30, 40),
            "longitude": np.arange(-100, -90),
            "CMAQ": np.random.rand(10),
            "Obs": np.random.rand(10),
        }
    )

    # 1. Create a figure and a cartopy GeoAxes
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})

    # 2. Instantiate the plot on the existing axes
    # This will fail if the __init__ refactor was incorrect
    plot = SpatialBiasScatterPlot(df, col1="Obs", col2="CMAQ", ax=ax)

    # 3. Assert that the plot object is using the correct axes
    assert plot.ax is ax

    # 4. Call the plot method
    plot.plot()

    # 5. Assert that a scatter plot was actually created
    # A scatter plot adds a PathCollection to the axes
    assert len(ax.collections) > 0


def test_spatial_bias_scatter_honors_constructor_limits_for_scale_and_sizes():
    """Constructor vmin/vmax should control color limits and size scaling."""
    df = pd.DataFrame(
        {
            "latitude": [30.0, 31.0, 32.0],
            "longitude": [-100.0, -99.0, -98.0],
            "Obs": [0.0, 0.0, 0.0],
            "CMAQ": [1.0, 2.0, 3.0],
        }
    )

    plot = SpatialBiasScatterPlot(df, col1="Obs", col2="CMAQ", vmin=-20, vmax=20, fact=1)
    ax = plot.plot()

    scatter = next(
        coll for coll in ax.collections if isinstance(coll, mcoll.PathCollection)
    )

    assert scatter.norm.boundaries[0] == -20
    assert scatter.norm.boundaries[-1] == 20
    assert np.allclose(scatter.get_sizes(), np.array([5.0, 10.0, 15.0]))
