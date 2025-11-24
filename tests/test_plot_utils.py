import pytest
import numpy as np
import pandas as pd
import xarray as xr
from monet_plots import plot_utils


def test_to_dataframe():
    """Test the to_dataframe function."""
    df = pd.DataFrame({'a': [1, 2, 3]})
    assert plot_utils.to_dataframe(df) is df

    da = xr.DataArray(np.random.rand(3, 3), name='data')
    assert isinstance(plot_utils.to_dataframe(da), pd.DataFrame)

    arr = np.random.rand(3, 3)
    assert isinstance(plot_utils.to_dataframe(arr), pd.DataFrame)

    with pytest.raises(TypeError):
        plot_utils.to_dataframe(1)


def test_validate_dataframe():
    """Test the validate_dataframe function."""
    df = pd.DataFrame({'a': [1, 2, 3]})
    plot_utils.validate_dataframe(df, required_columns=['a'])

    with pytest.raises(ValueError):
        plot_utils.validate_dataframe(df, required_columns=['b'])

    with pytest.raises(ValueError):
        plot_utils.validate_dataframe(pd.DataFrame(), required_columns=['a'])


def test_validate_plot_parameters():
    """Test the validate_plot_parameters function."""
    plot_utils.validate_plot_parameters('SpatialPlot', 'plot', discrete=True, ncolors=10, plotargs={'cmap': 'viridis'})

    with pytest.raises(TypeError):
        plot_utils.validate_plot_parameters('SpatialPlot', 'plot', discrete='true')

    with pytest.raises(ValueError):
        plot_utils.validate_plot_parameters('SpatialPlot', 'plot', ncolors=0)


def test_validate_data_array():
    """Test the validate_data_array function."""
    da = xr.DataArray(np.random.rand(3, 3), dims=['x', 'y'])
    plot_utils.validate_data_array(da, required_dims=['x', 'y'])

    with pytest.raises(ValueError):
        plot_utils.validate_data_array(da, required_dims=['z'])


def test_dynamic_fig_size():
    """Test the _dynamic_fig_size function."""
    da = xr.DataArray(np.random.rand(10, 20), dims=['y', 'x'])
    width, height = plot_utils._dynamic_fig_size(da)
    assert width > 0
    assert height > 0


def test_set_outline_patch_alpha():
    """Test the _set_outline_patch_alpha function."""
    from unittest.mock import MagicMock
    ax = MagicMock()
    plot_utils._set_outline_patch_alpha(ax, 0)
    ax.axes.outline_patch.set_alpha.assert_called_with(0)