import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from monet_plots.plots.sp_scatter_bias import SpScatterBiasPlot


@pytest.fixture
def clear_figures():
    """Clear all existing figures before and after a test."""
    plt.close("all")
    yield
    plt.close("all")


@pytest.fixture
def sample_data():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "latitude": np.random.uniform(20, 50, 10),
            "longitude": np.random.uniform(-120, -70, 10),
            "col1": np.random.rand(10),
            "col2": np.random.rand(10),
        }
    )


def test_sp_scatter_bias_plot_basic_creation(clear_figures, sample_data):
    """Test basic creation of SpScatterBiasPlot."""
    plot = SpScatterBiasPlot(df=sample_data, col1="col1", col2="col2")
    ax = plot.plot()
    assert ax is not None
    # The scatter plot itself is a collection
    assert len(ax.collections) >= 1


def test_sp_scatter_bias_plot_with_map_features(clear_figures, sample_data):
    """Test that map features are correctly added to the plot."""
    # The scatter plot creates one collection. Adding states should add more.
    plot = SpScatterBiasPlot(df=sample_data, col1="col1", col2="col2", states=True)
    ax = plot.plot()
    assert ax is not None
    assert len(ax.collections) > 1, "Expected more than one collection when adding states"


def test_sp_scatter_bias_plot_custom_projection(clear_figures, sample_data):
    """Test using a custom cartopy projection."""
    projection = ccrs.LambertConformal()
    plot = SpScatterBiasPlot(
        df=sample_data, col1="col1", col2="col2", projection=projection
    )
    ax = plot.plot()
    assert ax is not None
    # Check if the ax projection matches the one we specified
    assert isinstance(ax.projection, ccrs.LambertConformal)
