import numpy as np
import datetime
from unittest.mock import MagicMock
from monet_plots.plots.spatial_contour import SpatialContourPlot


def test_spatial_contour_plot():
    """Test the SpatialContourPlot plot method."""
    # Create a mock basemap object
    mock_map = MagicMock()
    mock_map.return_value = (np.random.rand(10, 10), np.random.rand(10, 10))

    # Create a mock grid object
    mock_grid = MagicMock()
    mock_grid.variables = {
        "LAT": np.random.rand(1, 1, 10, 10),
        "LON": np.random.rand(1, 1, 10, 10),
    }

    # Create a sample model variable
    modelvar = np.random.rand(10, 10)

    # Create a SpatialContourPlot instance
    plot = SpatialContourPlot(modelvar, mock_grid, datetime.datetime.now(), ncolors=10)

    # Call the plot method
    c = plot.plot(cmap="viridis", levels=np.arange(0, 1.1, 0.1))

    # Assert that the plot objects are created
    assert c is not None


def test_spatial_contour_plot_no_date():
    """Test the SpatialContourPlot plot method without a date."""
    # Create a mock basemap object
    mock_map = MagicMock()
    mock_map.return_value = (np.random.rand(10, 10), np.random.rand(10, 10))

    # Create a mock grid object
    mock_grid = MagicMock()
    mock_grid.variables = {
        "LAT": np.random.rand(1, 1, 10, 10),
        "LON": np.random.rand(1, 1, 10, 10),
    }

    # Create a sample model variable
    modelvar = np.random.rand(10, 10)

    # Create a SpatialContourPlot instance
    plot = SpatialContourPlot(modelvar, mock_grid, ncolors=10)

    # Call the plot method
    c = plot.plot(cmap="viridis", levels=np.arange(0, 1.1, 0.1))

    # Assert that the plot objects are created
    assert c is not None


def test_spatial_contour_plot_continuous():
    """Test the SpatialContourPlot plot method with a continuous colorbar."""
    # Create a mock basemap object
    mock_map = MagicMock()
    mock_map.return_value = (np.random.rand(10, 10), np.random.rand(10, 10))

    # Create a mock grid object
    mock_grid = MagicMock()
    mock_grid.variables = {
        "LAT": np.random.rand(1, 1, 10, 10),
        "LON": np.random.rand(1, 1, 10, 10),
    }

    # Create a sample model variable
    modelvar = np.random.rand(10, 10)

    # Create a SpatialContourPlot instance
    plot = SpatialContourPlot(
        modelvar, mock_grid, datetime.datetime.now(), discrete=False, ncolors=10
    )

    # Call the plot method
    c = plot.plot(cmap="viridis", levels=np.arange(0, 1.1, 0.1))

    # Assert that the plot objects are created
    assert c is not None
