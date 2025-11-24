import pytest
import numpy as np
import pandas as pd
import datetime
from unittest.mock import MagicMock
from monet_plots.plots.spatial_bias_scatter import SpatialBiasScatterPlot


def test_spatial_bias_scatter_plot():
    """Test the SpatialBiasScatterPlot plot method."""
    # Create a mock basemap object
    mock_map = MagicMock()
    mock_map.return_value = (np.random.rand(10), np.random.rand(10))

    # Create a sample dataframe
    df = pd.DataFrame({
        'latitude': np.arange(30, 40),
        'longitude': np.arange(-100, -90),
        'CMAQ': np.random.rand(10),
        'Obs': np.random.rand(10),
        'datetime': [datetime.datetime(2020, 1, 1)] * 10
    })

    # Create a SpatialBiasScatterPlot instance
    plot = SpatialBiasScatterPlot(df, col1='Obs', col2='CMAQ')

    # Call the plot method
    ax = plot.plot()

    # Assert that the plot objects are created
    assert ax is not None