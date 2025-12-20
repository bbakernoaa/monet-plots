import pytest
import numpy as np
import xarray as xr
from monet_plots import cartopy_utils
import pandas as pd


@pytest.fixture
def sample_da():
    """Create a sample xarray.DataArray for testing."""
    return xr.DataArray(
        np.random.rand(10, 10),
        coords={"latitude": np.arange(30, 40), "longitude": np.arange(-100, -90)},
        dims=["latitude", "longitude"],
    )


@pytest.fixture
def sample_df():
    """Create a sample pandas.DataFrame for testing."""
    return pd.DataFrame(
        {
            "latitude": np.arange(30, 40),
            "longitude": np.arange(-100, -90),
            "value": np.random.rand(10),
        }
    )


def test_plot_quick_imshow(sample_da):
    """Test the plot_quick_imshow function."""
    fig, ax = cartopy_utils.plot_quick_imshow(sample_da)
    assert fig is not None
    assert ax is not None


def test_plot_quick_map(sample_da):
    """Test the plot_quick_map function."""
    fig, ax = cartopy_utils.plot_quick_map(sample_da)
    assert fig is not None
    assert ax is not None


def test_plot_quick_contourf(sample_da):
    """Test the plot_quick_contourf function."""
    fig, ax = cartopy_utils.plot_quick_contourf(sample_da)
    assert fig is not None
    assert ax is not None


def test_facet_time_map(sample_da):
    """Test the facet_time_map function."""
    da = sample_da.expand_dims("time", axis=0)
    da["time"] = pd.date_range("2000-01-01", periods=1)
    fig, axes = cartopy_utils.facet_time_map(da)
    assert fig is not None
    assert axes is not None


def test_plot_points_map(sample_df):
    """Test the plot_points_map function."""
    fig, ax = cartopy_utils.plot_points_map(sample_df)
    assert fig is not None
    assert ax is not None


def test_plot_lines_map(sample_df):
    """Test the plot_lines_map function."""
    fig, ax = cartopy_utils.plot_lines_map(sample_df)
    assert fig is not None
    assert ax is not None


def test_plot_quick_imshow_labels(sample_da):
    """Test the plot_quick_imshow function with labels."""
    fig, ax = cartopy_utils.plot_quick_imshow(
        sample_da,
        xlabel="xlabel",
        ylabel="ylabel",
        title="title",
        cbar_label="cbar_label",
        xticks=np.arange(-100, -89, 5),
        yticks=np.arange(30, 41, 5),
        annotations=[{"text": "A", "xy": (-95, 35)}],
    )
    assert fig is not None
    assert ax is not None
