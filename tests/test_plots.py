# tests/test_plots.py
import monet_plots
import pandas as pd
import numpy as np
import xarray as xr
import pytest
from unittest.mock import Mock
import matplotlib.pyplot as plt
import cartopy.feature as cfeature


def test_spatial_plot(cleanup_plots):
    """Tests the SpatialPlot class."""
    plot = monet_plots.SpatialPlot()
    assert plot.ax is not None
    
    # Check if cfeature.LAND is added to the plot
    land_feature_found = False
    for feature in plot.ax.features:
        if isinstance(feature, cfeature.NaturalEarthFeature) and feature.name == 'land':
            land_feature_found = True
            break
    assert land_feature_found, "cfeature.LAND was not added to the SpatialPlot."
    plot.close()


def test_timeseries_plot(cleanup_plots):
    """Tests the TimeSeriesPlot class."""
    data = {'time': pd.to_datetime(['2025-01', '2025-01-02']), 'obs': [1, 2]}
    df = pd.DataFrame(data)
    plot = monet_plots.TimeSeriesPlot()
    plot.plot(df)
    assert plot.ax is not None
    plot.close()


def test_taylor_diagram_plot(cleanup_plots):
    """Tests the TaylorDiagramPlot class."""
    obs = np.random.rand(10)
    model = obs + np.random.rand(10) * 0.1
    df = pd.DataFrame({'obs': obs, 'model': model})
    plot = monet_plots.TaylorDiagramPlot(obs.std())
    plot.add_sample(df)
    assert plot.dia is not None
    plot.close()


def test_kde_plot(cleanup_plots):
    """Tests the KDEPlot class."""
    data = np.random.randn(100)
    plot = monet_plots.KDEPlot()
    plot.plot(data)
    assert plot.ax is not None
    plot.close()


def test_scatter_plot(cleanup_plots):
    """Tests the ScatterPlot class."""
    data = {'x': np.arange(10), 'y': np.arange(10)}
    df = pd.DataFrame(data)
    plot = monet_plots.ScatterPlot()
    plot.plot(df, 'x', 'y')
    assert plot.ax is not None
    plot.close()


def test_facet_grid_plot_pandas(cleanup_plots):
    """Tests the FacetGridPlot class with pandas DataFrame."""
    # Create a pandas DataFrame for the test
    data = pd.DataFrame({
        'x': np.tile([1, 2], 6),
        'y': np.repeat([1, 2, 3], 4),
        'z': np.tile([1, 2, 3, 4], 3),
        'value': np.random.randn(12)
    })
    plot = monet_plots.FacetGridPlot(data, col='z')
    assert plot.grid is not None
    plot.close()


def test_facet_grid_plot_xarray(cleanup_plots, mock_data_factory):
    """Tests the FacetGridPlot class with xarray DataArray."""
    # Test with xarray DataArray that should be converted to DataFrame
    xarray_data = mock_data_factory.facet_data()
    plot = monet_plots.FacetGridPlot(xarray_data, col='time')
    assert plot.grid is not None
    plot.close()


def test_wind_quiver_plot_with_gridobj(cleanup_plots, mock_grid_object):
    """Tests the WindQuiverPlot class with grid object."""
    plot = monet_plots.WindQuiverPlot()
    
    # Create sample wind data
    ws = np.random.uniform(0, 20, (10, 10))  # wind speed
    wdir = np.random.uniform(0, 360, (10, 10))  # wind direction
    
    # Test with wind speed and direction
    plot.plot(ws=ws, wdir=wdir, gridobj=mock_grid_object)
    assert plot.ax is not None
    plot.close()


def test_wind_barbs_plot_with_gridobj(cleanup_plots, mock_grid_object):
    """Tests the WindBarbsPlot class with grid object."""
    plot = monet_plots.WindBarbsPlot()
    
    # Create sample wind data
    ws = np.random.uniform(0, 20, (10, 10))  # wind speed
    wdir = np.random.uniform(0, 360, (10, 10))  # wind direction
    
    # Test with wind speed and direction
    plot.plot(ws=ws, wdir=wdir, gridobj=mock_grid_object)
    assert plot.ax is not None
    plot.close()


def test_wind_quiver_plot_with_uv_components(cleanup_plots, mock_grid_object):
    """Tests the WindQuiverPlot class with u, v components."""
    plot = monet_plots.WindQuiverPlot()
    
    # Create sample wind data
    u = np.random.uniform(-10, 10, (10, 10))  # u component
    v = np.random.uniform(-10, 10, (10, 10))  # v component
    
    # Create coordinate data from grid object
    lat = mock_grid_object.variables['LAT'][0, 0, :, :].squeeze()
    lon = mock_grid_object.variables['LON'][0, 0, :, :].squeeze()
    
    # Test with u, v components
    plot.plot(u=u, v=v, x=lon, y=lat)
    assert plot.ax is not None
    plot.close()


def test_wind_barbs_plot_with_uv_components(cleanup_plots, mock_grid_object):
    """Tests the WindBarbsPlot class with u, v components."""
    plot = monet_plots.WindBarbsPlot()
    
    # Create sample wind data
    u = np.random.uniform(-10, 10, (10, 10))  # u component
    v = np.random.uniform(-10, 10, (10, 10))  # v component
    
    # Create coordinate data from grid object
    lat = mock_grid_object.variables['LAT'][0, 0, :, :].squeeze()
    lon = mock_grid_object.variables['LON'][0, 0, :, :].squeeze()
    
    # Test with u, v components
    plot.plot(u=u, v=v, x=lon, y=lat)
    assert plot.ax is not None
    plot.close()
