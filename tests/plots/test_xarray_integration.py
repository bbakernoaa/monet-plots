# tests/plots/test_xarray_integration.py

import pytest
import pandas as pd
import xarray as xr
import numpy as np

from monet_plots.plot_utils import normalize_data

from monet_plots.plots.timeseries import TimeSeriesPlot

# TDD: Test the normalization utility
def test_normalize_dataframe_passthrough():
    df = pd.DataFrame({'a': [1, 2]})
    assert normalize_data(df) is df

def test_normalize_dataarray_to_dataset():
    da = xr.DataArray(np.random.rand(3), name='test_var')
    ds = normalize_data(da)
    assert isinstance(ds, xr.Dataset)
    assert 'test_var' in ds.data_vars

def test_normalize_dataset_passthrough():
    ds = xr.Dataset({'a': ('x', [1, 2, 3])})
    assert normalize_data(ds) is ds

def test_normalize_invalid_type():
    with pytest.raises(TypeError):
        normalize_data(np.array([1, 2, 3]))

from monet_plots.plots.timeseries import TimeSeriesPlot, TimeSeriesStatsPlot

# TDD: Test the refactored TimeSeriesPlot
@pytest.fixture
def sample_xarray_dataset():
    # Create a sample xarray Dataset for testing
    time = pd.to_datetime(['2023-01-01T00:00', '2023-01-01T12:00', '2023-01-02T00:00', '2023-01-02T12:00'])
    temp_data = np.array([10, 20, 5, 15])
    model_data = np.array([12, 18, 7, 13])
    return xr.Dataset(
        {
            'temperature': (('time',), temp_data),
            'model': (('time',), model_data)
        },
        coords={'time': time}
    )

@pytest.fixture
def sample_pandas_dataframe():
    # Create a sample pandas DataFrame for testing
    return pd.DataFrame({
        'time': pd.to_datetime(['2023-01-01T00:00', '2023-01-01T12:00', '2023-01-02T00:00', '2023-01-02T12:00']),
        'temperature': np.array([10, 20, 5, 15]),
        'model': np.array([12, 18, 7, 13]),
        'units': 'K'
    })

def test_timeseries_plot_with_xarray(sample_xarray_dataset):
    """
    Verify that TimeSeriesPlot can plot directly from an xarray Dataset.
    """
    plot = TimeSeriesPlot(sample_xarray_dataset, y='temperature', freq='D')
    ax = plot.plot()
    assert ax is not None
    line = ax.get_lines()[0]
    _, y_data = line.get_data()
    assert np.isclose(y_data[0], 15) # Mean of 10, 20
    assert np.isclose(y_data[1], 10) # Mean of 5, 15

def test_timeseries_plot_with_pandas(sample_pandas_dataframe):
    """
    Ensure that TimeSeriesPlot still works perfectly with pandas DataFrames.
    """
    plot = TimeSeriesPlot(sample_pandas_dataframe, y='temperature', freq='D')
    ax = plot.plot()
    assert ax is not None
    line = ax.get_lines()[0]
    _, y_data = line.get_data()
    assert np.isclose(y_data[0], 15) # Mean of 10, 20
    assert np.isclose(y_data[1], 10) # Mean of 5, 15

def test_timeseries_plot_hourly_freq(sample_xarray_dataset):
    """
    Test the `freq` parameter with a different frequency.
    """
    plot = TimeSeriesPlot(sample_xarray_dataset, y='temperature', freq='12H')
    ax = plot.plot()
    assert ax is not None
    line = ax.get_lines()[0]
    _, y_data = line.get_data()
    assert len(y_data) == 4

# TDD: Test the refactored TimeSeriesStatsPlot
def test_stats_plot_bias_with_xarray(sample_xarray_dataset):
    """
    Test bias calculation with xarray data.
    """
    plot = TimeSeriesStatsPlot(sample_xarray_dataset, col1='temperature', col2='model')
    ax = plot.plot(stat='bias', freq='D')
    assert ax is not None
    line = ax.get_lines()[0]
    _, y_data = line.get_data()
    assert np.isclose(y_data[0], 0)
    assert np.isclose(y_data[1], 0)

def test_stats_plot_rmse_with_pandas(sample_pandas_dataframe):
    """
    Test RMSE calculation with pandas data.
    """
    plot = TimeSeriesStatsPlot(sample_pandas_dataframe, col1='temperature', col2='model')
    ax = plot.plot(stat='rmse', freq='D')
    assert ax is not None
    line = ax.get_lines()[0]
    _, y_data = line.get_data()
    assert np.isclose(y_data[0], np.sqrt(4))
    assert np.isclose(y_data[1], np.sqrt(4))
