from unittest.mock import MagicMock

import cartopy.crs as ccrs
import pytest

from monet_plots.plots.spatial import SpatialPlot


@pytest.fixture
def spatial_plot():
    """Returns a SpatialPlot instance with no features for testing."""
    return SpatialPlot(projection=ccrs.PlateCarree(), coastlines=False)


def test_add_features_boolean(spatial_plot):
    """Test that features are added with boolean flags."""
    # The plot is initialized with no features, so the list should be empty.
    assert len(spatial_plot.ax.collections) == 0

    # Add features using boolean flags
    spatial_plot.add_features(coastlines=True, states=True, ocean=True)

    # Check that three features have been added
    assert len(spatial_plot.ax.collections) == 3


def test_add_features_dict(spatial_plot):
    """Test that features are styled with dictionaries."""
    # The plot is initialized with no features, so the list should be empty.
    assert len(spatial_plot.ax.collections) == 0

    # Add a feature with a style dictionary
    style = {"linewidth": 2, "edgecolor": "red"}
    spatial_plot.add_features(states=style)

    # Check that one feature has been added
    assert len(spatial_plot.ax.collections) == 1

    # Check that the properties of the added collection are correct
    added_collection = spatial_plot.ax.collections[0]
    assert added_collection.get_linewidth()[0] == style["linewidth"]


def test_add_features_gridlines(spatial_plot):
    """Test that gridlines are added correctly."""
    # Mock the ax.gridlines method to check if it's called
    spatial_plot.ax.gridlines = MagicMock()

    # Call add_features with gridlines=True
    spatial_plot.add_features(gridlines=True)

    # Check that ax.gridlines was called
    spatial_plot.ax.gridlines.assert_called_once()


def test_add_features_natural_earth(spatial_plot):
    """Test the natural_earth convenience flag."""
    # The plot is initialized with no features, so the list should be empty.
    assert len(spatial_plot.ax.collections) == 0

    # Use the natural_earth flag
    spatial_plot.add_features(natural_earth=True)

    # Check that the standard set of features has been added
    # (ocean, land, lakes, rivers)
    assert len(spatial_plot.ax.collections) == 4
