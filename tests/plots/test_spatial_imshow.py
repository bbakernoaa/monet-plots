import pytest
import numpy as np
from unittest.mock import MagicMock
from monet_plots.plots.spatial_imshow import SpatialImshow


@pytest.fixture
def mock_grid():
    """Create a mock grid object."""
    grid = MagicMock()
    grid.variables = {
        'LAT': np.random.uniform(low=20, high=50, size=(1, 1, 10, 10)),
        'LON': np.random.uniform(low=-120, high=-70, size=(1, 1, 10, 10))
    }
    return grid


def test_spatial_imshow_plot(mock_grid):
    """Test the SpatialImshow plot method."""
    # Create a sample model variable
    modelvar = np.random.rand(10, 10)

    # Create a SpatialImshow instance
    plot = SpatialImshow(modelvar, mock_grid)

    # Call the plot method
    c = plot.plot()

    # Assert that the plot objects are created
    assert plot.fig is not None
    assert plot.ax is not None
    assert c is not None


def test_spatial_imshow_plot_discrete(mock_grid):
    """Test the SpatialImshow plot method with a discrete colorbar."""
    # Create a sample model variable
    modelvar = np.random.rand(10, 10)

    # Create a SpatialImshow instance
    plot = SpatialImshow(modelvar, mock_grid, discrete=True)

    # Call the plot method
    c = plot.plot()

    # Assert that the plot objects are created
    assert plot.fig is not None
    assert plot.ax is not None
    assert c is not None


def test_spatial_imshow_plot_discrete_vmin_vmax(mock_grid):
    """Test the SpatialImshow plot method with a discrete colorbar and vmin/vmax."""
    # Create a sample model variable
    modelvar = np.random.rand(10, 10)

    # Create a SpatialImshow instance
    plot = SpatialImshow(modelvar, mock_grid, discrete=True,
                         plotargs={'vmin': 0.1, 'vmax': 0.9})

    # Call the plot method
    c = plot.plot()

    # Assert that the plot objects are created
    assert plot.fig is not None
    assert plot.ax is not None
    assert c is not None
