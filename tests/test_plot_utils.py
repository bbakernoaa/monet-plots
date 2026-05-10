import numpy as np
import pandas as pd
import pytest
import xarray as xr

from monet_plots import plot_utils


def test_to_dataframe():
    """Test the to_dataframe function."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert plot_utils.to_dataframe(df) is df

    da = xr.DataArray(np.random.rand(3, 3), name="data")
    assert isinstance(plot_utils.to_dataframe(da), pd.DataFrame)

    arr = np.random.rand(3, 3)
    assert isinstance(plot_utils.to_dataframe(arr), pd.DataFrame)

    with pytest.raises(TypeError):
        plot_utils.to_dataframe(1)

    # Test 1D and 2D numpy arrays
    df1 = plot_utils.to_dataframe(np.array([1, 2, 3]))
    assert "col_0" in df1.columns
    assert len(df1) == 3

    df2 = plot_utils.to_dataframe(np.array([[1, 2], [3, 4]]))
    assert "col_0" in df2.columns
    assert "col_1" in df2.columns
    assert len(df2) == 2

    # Test numpy array with > 2 dims
    with pytest.raises(ValueError, match="dimensions not supported"):
        plot_utils.to_dataframe(np.zeros((2, 2, 2)))


def test_validate_dataframe():
    """Test the validate_dataframe function."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    plot_utils.validate_dataframe(df, required_columns=["a"])

    with pytest.raises(ValueError):
        plot_utils.validate_dataframe(df, required_columns=["b"])

    with pytest.raises(ValueError):
        plot_utils.validate_dataframe(pd.DataFrame(), required_columns=["a"])


def test_validate_plot_parameters():
    """Test the validate_plot_parameters function."""
    plot_utils.validate_plot_parameters(
        "SpatialPlot", "plot", discrete=True, ncolors=10, plotargs={"cmap": "viridis"}
    )

    with pytest.raises(TypeError, match="discrete parameter must be boolean"):
        plot_utils.validate_plot_parameters("SpatialPlot", "plot", discrete="true")

    with pytest.raises(TypeError, match="ncolors parameter must be integer"):
        plot_utils.validate_plot_parameters("SpatialPlot", "plot", ncolors=10.5)

    with pytest.raises(ValueError, match="ncolors parameter must be between"):
        plot_utils.validate_plot_parameters("SpatialPlot", "plot", ncolors=0)

    with pytest.raises(TypeError, match="plotargs parameter must be dict"):
        plot_utils.validate_plot_parameters("SpatialPlot", "plot", plotargs="not a dict")

    with pytest.raises(TypeError, match="colormap must be string"):
        plot_utils.validate_plot_parameters(
            "SpatialPlot", "plot", plotargs={"cmap": 123}
        )

    # TimeSeriesPlot validation
    plot_utils.validate_plot_parameters(
        "TimeSeriesPlot", "plot", x="time", y="obs", fillargs={"alpha": 0.5}
    )

    with pytest.raises(TypeError, match="x parameter must be string"):
        plot_utils.validate_plot_parameters("TimeSeriesPlot", "plot", x=123)

    with pytest.raises(TypeError, match="y parameter must be string"):
        plot_utils.validate_plot_parameters("TimeSeriesPlot", "plot", y=123)

    with pytest.raises(TypeError, match="fillargs parameter must be dict"):
        plot_utils.validate_plot_parameters(
            "TimeSeriesPlot", "plot", fillargs="not a dict"
        )

    with pytest.raises(TypeError, match="alpha must be numeric"):
        plot_utils.validate_plot_parameters(
            "TimeSeriesPlot", "plot", fillargs={"alpha": "high"}
        )

    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        plot_utils.validate_plot_parameters(
            "TimeSeriesPlot", "plot", fillargs={"alpha": 1.5}
        )


def test_validate_data_array():
    """Test the validate_data_array function."""
    with pytest.raises(ValueError, match="data cannot be None"):
        plot_utils.validate_data_array(None)

    with pytest.raises(TypeError, match="data must have a shape attribute"):
        plot_utils.validate_data_array(123)

    da = xr.DataArray(np.random.rand(3, 3), dims=["x", "y"])
    plot_utils.validate_data_array(da, required_dims=["x", "y"])

    with pytest.raises(ValueError):
        plot_utils.validate_data_array(da, required_dims=["z"])

    with pytest.raises(TypeError, match="data must have dims attribute"):
        plot_utils.validate_data_array(np.zeros((3, 3)), required_dims=["x"])


def test_dynamic_fig_size():
    """Test the _dynamic_fig_size function."""
    da = xr.DataArray(np.random.rand(10, 20), dims=["y", "x"])
    width, height = plot_utils._dynamic_fig_size(da)
    assert width > 0
    assert height > 0

    da2 = xr.DataArray(
        np.random.rand(10, 20),
        dims=["latitude", "longitude"],
        coords={"latitude": np.arange(10), "longitude": np.arange(20)},
    )
    assert plot_utils._dynamic_fig_size(da2) == (10, 10 * 0.5)

    da3 = xr.DataArray(
        np.random.rand(10, 20),
        dims=["lat", "lon"],
        coords={"lat": np.arange(10), "lon": np.arange(20)},
    )
    assert plot_utils._dynamic_fig_size(da3) == (10, 10 * 0.5)


def test_set_outline_patch_alpha():
    """Test the _set_outline_patch_alpha function."""
    from unittest.mock import MagicMock

    ax = MagicMock()
    plot_utils._set_outline_patch_alpha(ax, 0)
    ax.axes.outline_patch.set_alpha.assert_called_with(0)
