#!/usr/bin/env python3
"""
Test script to verify xarray integration in monet_plots.

This script tests that the TimeSeriesPlot class can handle both pandas DataFrames
and xarray DataArrays/Datasets without forcing conversion to pandas.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Check if xarray is available
try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False
    print("xarray not available, skipping xarray tests")

from monet_plots.plots.timeseries import TimeSeriesPlot
from monet_plots.plot_utils import normalize_data

def test_normalize_data():
    """Test the normalize_data function."""
    print("Testing normalize_data function...")

    # Test with pandas DataFrame
    df = pd.DataFrame({
        'time': pd.date_range('2023-01-01', periods=10),
        'value': np.random.randn(10)
    })
    result = normalize_data(df)
    print(f"✓ Pandas DataFrame normalized: {type(result)}")
    assert isinstance(result, pd.DataFrame)

    # Test with numpy array
    arr = np.random.randn(10, 2)
    result = normalize_data(arr)
    print(f"✓ Numpy array normalized: {type(result)}")
    assert isinstance(result, pd.DataFrame)

    if HAS_XARRAY:
        # Test with xarray DataArray
        da = xr.DataArray(
            np.random.randn(10),
            dims=['time'],
            coords={'time': pd.date_range('2023-01-01', periods=10)}
        )
        result = normalize_data(da)
        print(f"✓ Xarray DataArray normalized: {type(result)}")
        assert isinstance(result, xr.DataArray)

        # Test with xarray Dataset
        ds = xr.Dataset({
            'value': (['time'], np.random.randn(10)),
            'other': (['time'], np.random.randn(10))
        }, coords={'time': pd.date_range('2023-01-01', periods=10)})
        result = normalize_data(ds)
        print(f"✓ Xarray Dataset normalized: {type(result)}")
        assert isinstance(result, xr.Dataset)

    print("All normalize_data tests passed!\n")

def test_timeseries_plot_pandas():
    """Test TimeSeriesPlot with pandas DataFrame."""
    print("Testing TimeSeriesPlot with pandas DataFrame...")

    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='H')
    df = pd.DataFrame({
        'time': dates,
        'obs': np.random.normal(0, 1, 100),
        'model': np.random.normal(0.1, 1.1, 100)
    })

    # Create and plot
    plot = TimeSeriesPlot(df, x='time', y='obs', title="Pandas TimeSeries Test")
    ax = plot.plot()

    # Save the plot
    plot.save("test_pandas_timeseries.png")
    print("✓ Pandas TimeSeries plot created and saved")

    # Clean up
    plot.close()
    print("✓ Pandas TimeSeries plot closed\n")

def test_timeseries_plot_xarray():
    """Test TimeSeriesPlot with xarray DataArray."""
    if not HAS_XARRAY:
        print("Skipping xarray TimeSeriesPlot test (xarray not available)\n")
        return

    print("Testing TimeSeriesPlot with xarray DataArray...")

    # Create sample xarray data
    dates = pd.date_range('2023-01-01', periods=100, freq='H')
    da = xr.DataArray(
        np.random.normal(0, 1, 100),
        dims=['time'],
        coords={'time': dates},
        name='obs'
    )

    # Create and plot
    plot = TimeSeriesPlot(da, x='time', y='obs', title="Xarray TimeSeries Test")
    ax = plot.plot()

    # Save the plot
    plot.save("test_xarray_timeseries.png")
    print("✓ Xarray TimeSeries plot created and saved")

    # Clean up
    plot.close()
    print("✓ Xarray TimeSeries plot closed\n")

def test_timeseries_plot_xarray_dataset():
    """Test TimeSeriesPlot with xarray Dataset."""
    if not HAS_XARRAY:
        print("Skipping xarray Dataset TimeSeriesPlot test (xarray not available)\n")
        return

    print("Testing TimeSeriesPlot with xarray Dataset...")

    # Create sample xarray dataset
    dates = pd.date_range('2023-01-01', periods=100, freq='H')
    ds = xr.Dataset({
        'obs': (['time'], np.random.normal(0, 1, 100)),
        'model': (['time'], np.random.normal(0.1, 1.1, 100))
    }, coords={'time': dates})

    # Create and plot
    plot = TimeSeriesPlot(ds, x='time', y='obs', title="Xarray Dataset TimeSeries Test")
    ax = plot.plot()

    # Save the plot
    plot.save("test_xarray_dataset_timeseries.png")
    print("✓ Xarray Dataset TimeSeries plot created and saved")

    # Clean up
    plot.close()
    print("✓ Xarray Dataset TimeSeries plot closed\n")

if __name__ == "__main__":
    print("Starting xarray integration tests...\n")

    # Run tests
    test_normalize_data()
    test_timeseries_plot_pandas()
    test_timeseries_plot_xarray()
    test_timeseries_plot_xarray_dataset()

    print("All tests completed successfully!")