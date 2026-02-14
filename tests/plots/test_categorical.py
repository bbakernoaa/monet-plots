import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from monet_plots.plots.categorical import categorical_plot


@pytest.fixture
def clear_figures():
    """Clear all existing figures before and after a test."""
    plt.close("all")
    yield
    plt.close("all")


@pytest.fixture
def sample_da():
    """Create a sample DataArray for testing."""
    return xr.DataArray(
        np.random.rand(10, 3),
        dims=("instance", "category"),
        coords={
            "instance": np.arange(10),
            "category": ["A", "B", "C"],
            "site": ("instance", ["site1"] * 5 + ["site2"] * 5),
        },
    )


def test_categorical_plot_bar(clear_figures, sample_da):
    """Test that categorical_plot creates a bar plot."""
    fig, ax = categorical_plot(
        sample_da, x="category", y=sample_da.name or "value", kind="bar"
    )
    assert fig is not None
    assert ax is not None
    # Check if there are bars in the plot
    assert len(ax.flatten()[0].patches) > 0


def test_categorical_plot_violin(clear_figures, sample_da):
    """Test that categorical_plot creates a violin plot."""
    fig, ax = categorical_plot(
        sample_da, x="category", y=sample_da.name or "value", kind="violin"
    )
    assert fig is not None
    assert ax is not None
    # Check if there are violins in the plot
    assert len(ax.flatten()[0].collections) > 0


def test_categorical_plot_hue(clear_figures, sample_da):
    """Test that categorical_plot works with hue."""
    fig, ax = categorical_plot(
        sample_da, x="category", y=sample_da.name or "value", hue="site", kind="bar"
    )
    assert fig is not None
    assert ax is not None
    assert len(ax.flatten()[0].patches) > 0
    # Check for legend
    assert len(fig.legends) > 0


def test_categorical_plot_missing_x_y_kwargs(clear_figures, sample_da):
    """Test that categorical_plot raises ValueError if x or y is not provided."""
    with pytest.raises(ValueError):
        categorical_plot(sample_da, kind="bar")
    with pytest.raises(ValueError):
        categorical_plot(sample_da, x="category", kind="bar")
    with pytest.raises(ValueError):
        categorical_plot(sample_da, y=sample_da.name, kind="bar")
