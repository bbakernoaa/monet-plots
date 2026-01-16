import matplotlib.pyplot as plt
import pandas as pd
import pytest

from monet_plots.plots.sp_scatter_bias import SpScatterBiasPlot


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Create a sample DataFrame for testing."""
    data = {
        "latitude": [34.0, 35.0, 36.0, 37.0, 38.0],
        "longitude": [-118.0, -119.0, -120.0, -121.0, -122.0],
        "obs": [10.0, 12.0, 15.0, 11.0, 13.0],
        "model": [11.0, 11.5, 16.0, 12.5, 12.0],
    }
    return pd.DataFrame(data)


def test_sp_scatter_bias_plot_from_dataframe(sample_dataframe: pd.DataFrame):
    """Test creating a SpScatterBiasPlot using the from_dataframe factory."""
    # The UI: A simple, non-interactive plot for validation.
    plot_instance = SpScatterBiasPlot.from_dataframe(
        df=sample_dataframe,
        col1="obs",
        col2="model",
        map_kwargs={"states": True},
    )

    # The Proof: Assert that the plot object is of the correct type
    # and the axes are properly configured.
    assert isinstance(plot_instance, SpScatterBiasPlot)
    assert plot_instance.ax is not None
    assert isinstance(plot_instance.ax, plt.Axes)

    # Check that data has been plotted by verifying collections were added.
    # A scatter plot adds a PathCollection.
    assert len(plot_instance.ax.collections) > 0, "Scatter plot should add a collection."

    # The CLI Command to run this test:
    # python -m pytest tests/plots/test_sp_scatter_bias.py
