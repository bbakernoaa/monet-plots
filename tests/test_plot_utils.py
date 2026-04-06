import pytest
import numpy as np
import pandas as pd
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

    with pytest.raises(TypeError):
        plot_utils.validate_plot_parameters("SpatialPlot", "plot", discrete="true")

    with pytest.raises(ValueError):
        plot_utils.validate_plot_parameters("SpatialPlot", "plot", ncolors=0)


def test_validate_data_array():
    """Test the validate_data_array function."""
    da = xr.DataArray(np.random.rand(3, 3), dims=["x", "y"])
    plot_utils.validate_data_array(da, required_dims=["x", "y"])

    with pytest.raises(ValueError):
        plot_utils.validate_data_array(da, required_dims=["z"])


def test_dynamic_fig_size():
    """Test the _dynamic_fig_size function."""
    da = xr.DataArray(np.random.rand(10, 20), dims=["y", "x"])
    width, height = plot_utils._dynamic_fig_size(da)
    assert width > 0
    assert height > 0


def test_set_outline_patch_alpha():
    """Test the _set_outline_patch_alpha function."""
    from unittest.mock import MagicMock

    ax = MagicMock()
    plot_utils._set_outline_patch_alpha(ax, 0)
    ax.axes.outline_patch.set_alpha.assert_called_with(0)


def test_is_lazy():
    """Test the is_lazy, is_dask, and is_cubed functions."""
    # Eager data
    assert not plot_utils.is_lazy(np.array([1, 2, 3]))
    assert not plot_utils.is_lazy(pd.Series([1, 2, 3]))
    assert not plot_utils.is_lazy(None)

    # Dask data
    try:
        import dask.array as da

        d_arr = da.from_array(np.array([1, 2, 3]), chunks=2)
        assert plot_utils.is_dask(d_arr)
        assert plot_utils.is_lazy(d_arr)

        # Xarray wrapped dask
        da_xr = xr.DataArray(d_arr)
        assert plot_utils.is_dask(da_xr)
        assert plot_utils.is_lazy(da_xr)
    except ImportError:
        pytest.skip("dask not installed")


def test_compute():
    """Test the compute function."""
    # Eager data
    arr = np.array([1, 2, 3])
    assert plot_utils.compute(arr) is arr

    # Multiple eager
    a, b = plot_utils.compute(1, 2)
    assert a == 1
    assert b == 2

    # Dask data
    try:
        import dask.array as da

        d_arr1 = da.from_array(np.array([1, 2, 3]), chunks=2)
        d_arr2 = da.from_array(np.array([4, 5, 6]), chunks=2)

        # Single compute
        res = plot_utils.compute(d_arr1)
        assert isinstance(res, np.ndarray)
        np.testing.assert_array_equal(res, [1, 2, 3])

        # Multiple compute
        res1, res2 = plot_utils.compute(d_arr1, d_arr2)
        assert isinstance(res1, np.ndarray)
        assert isinstance(res2, np.ndarray)
        np.testing.assert_array_equal(res1, [1, 2, 3])
        np.testing.assert_array_equal(res2, [4, 5, 6])

        # Mixed xarray/dask
        xr_arr = xr.DataArray(d_arr1, name="test")
        res_xr = plot_utils.compute(xr_arr)
        assert isinstance(res_xr, xr.DataArray)
        assert not res_xr.chunks
        np.testing.assert_array_equal(res_xr.values, [1, 2, 3])

    except ImportError:
        pytest.skip("dask not installed")
