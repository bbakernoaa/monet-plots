import datetime
from unittest.mock import MagicMock

import cartopy.crs as ccrs
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


def test_bias_scale_respects_shared_limits_and_shows_zero():
    data = pd.DataFrame(
        {
            "lat": [30, 31, 32, 33],
            "lon": [-99, -98, -97, -96],
            "obs": [0, 0, 0, 0],
            "model": [-0.05, 0, 0.3, np.nan],
        }
    )
    plot = SpatialBiasScatterPlot(
        data, "obs", "model", vmin=-0.1, vmax=0.1, coastlines=False
    )
    try:
        ax = plot.plot()
        marker = ax.collections[0]
        assert marker.norm.vmin == -0.1
        assert marker.norm.vmax == 0.1
        assert marker.get_sizes()[1] > 0
    finally:
        plot.close()
