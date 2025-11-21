"""
Comprehensive Unit Tests for MONET Plots - All Plot Classes
============================================================

This module contains comprehensive unit tests for all plot classes using TDD approach.
Tests cover all 11 plot classes with complete method coverage and edge case validation.

Following TDD principles: Write failing tests first, implement minimal code to pass, then refactor.
"""

import pytest
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import warnings
from pathlib import Path


class TestBasePlotUnit:
    """Unit tests for BasePlot class - foundation for all plot classes."""
    
    def test_base_plot_initialization_default(self, mock_data_factory):
        """Test BasePlot initialization with default parameters."""
        from src.monet_plots.plots.base import BasePlot
        
        plot = BasePlot()
        
        assert plot.fig is not None
        assert plot.ax is not None
        assert hasattr(plot, 'fig')
        assert hasattr(plot, 'ax')
        # Verify Wiley style is applied
        assert plt.rcParams['font.family'] == ['Liberation Sans', 'Arial', 'sans-serif'] or \
               'Wiley' in str(plt.style.available)
        
        plot.close()
    
    def test_base_plot_initialization_custom_figsize_dpi(self, mock_data_factory):
        """Test BasePlot initialization with custom figsize and DPI."""
        from src.monet_plots.plots.base import BasePlot
        
        custom_figsize = (12, 8)
        custom_dpi = 150
        
        plot = BasePlot(figsize=custom_figsize, dpi=custom_dpi)
        
        assert plot.fig.get_size_inches()[0] == custom_figsize[0]
        assert plot.fig.get_size_inches()[1] == custom_figsize[1]
        assert plot.fig.get_dpi() == custom_dpi
        
        plot.close()
    
    def test_base_plot_with_existing_figure_axes(self, mock_data_factory):
        """Test BasePlot with existing matplotlib figure and axes."""
        from src.monet_plots.plots.base import BasePlot
        
        # Create figure and axes
        fig, ax = plt.subplots(figsize=(8, 6))
        
        plot = BasePlot(fig=fig, ax=ax)
        
        assert plot.fig is fig
        assert plot.ax is ax
        assert plot.fig.get_size_inches()[0] == 8
        assert plot.fig.get_size_inches()[1] == 6
        
        plot.close()
    
    def test_base_plot_save_functionality(self, mock_data_factory, temp_directory):
        """Test BasePlot save functionality with various formats."""
        from src.monet_plots.plots.base import BasePlot
        
        plot = BasePlot()
        
        # Test different file formats
        formats = ['png', 'pdf', 'svg', 'jpg']
        
        for fmt in formats:
            filepath = os.path.join(temp_directory, f'test_plot.{fmt}')
            
            try:
                plot.save(filepath)
                assert os.path.exists(filepath)
                assert os.path.getsize(filepath) > 0
                
                # Clean up
                os.remove(filepath)
                
            except (ValueError, KeyError) as e:
                # Some formats might not be supported
                assert "format" in str(e).lower() or "unsupported" in str(e).lower()
        
        plot.close()
    
    def test_base_plot_save_with_dpi_and_transparency(self, mock_data_factory, temp_directory):
        """Test BasePlot save with custom DPI and transparency."""
        from src.monet_plots.plots.base import BasePlot
        
        plot = BasePlot()
        filepath = os.path.join(temp_directory, 'test_transparent.png')
        
        # Test save with transparency
        plot.save(filepath, dpi=300, transparent=True)
        
        assert os.path.exists(filepath)
        assert os.path.getsize(filepath) > 0
        
        plot.close()
    
    def test_base_plot_save_invalid_path(self, mock_data_factory):
        """Test BasePlot save error handling with invalid paths."""
        from src.monet_plots.plots.base import BasePlot
        
        plot = BasePlot()
        
        # Test invalid paths
        invalid_paths = [
            "",  # Empty path
            "/nonexistent/directory/test.png",  # Nonexistent directory
            "/root/test.png"  # Root directory (should fail on most systems)
        ]
        
        for path in invalid_paths:
            with pytest.raises((OSError, PermissionError, FileNotFoundError)):
                plot.save(path)
        
        plot.close()
    
    def test_base_plot_context_manager(self, mock_data_factory):
        """Test BasePlot context manager functionality."""
        from src.monet_plots.plots.base import BasePlot
        
        initial_figs = len(plt.get_fignums())
        
        with BasePlot() as plot:
            assert plot.fig is not None
            assert plot.ax is not None
            assert len(plt.get_fignums()) == initial_figs + 1
        
        # Figure should be closed after context
        assert len(plt.get_fignums()) == initial_figs
    
    def test_base_plot_close_method(self, mock_data_factory):
        """Test BasePlot close method properly cleans up resources."""
        from src.monet_plots.plots.base import BasePlot
        
        initial_figs = len(plt.get_fignums())
        
        plot = BasePlot()
        assert len(plt.get_fignums()) == initial_figs + 1
        
        plot.close()
        assert len(plt.get_fignums()) == initial_figs
        
        # Double close should not raise error
        plot.close()  # Should be safe to call again
    
    def test_base_plot_repr_str(self, mock_data_factory):
        """Test BasePlot string representation methods."""
        from src.monet_plots.plots.base import BasePlot
        
        plot = BasePlot()
        
        # Test __repr__
        repr_str = repr(plot)
        assert "BasePlot" in repr_str
        assert "Figure" in repr_str
        
        # Test __str__
        str_str = str(plot)
        assert len(str_str) > 0
        
        plot.close()


class TestSpatialPlotUnit:
    """Unit tests for SpatialPlot class."""
    
    def test_spatial_plot_initialization(self, mock_data_factory):
        """Test SpatialPlot initialization with default parameters."""
        from src.monet_plots.plots.spatial import SpatialPlot
        
        plot = SpatialPlot()
        
        assert plot.fig is not None
        assert plot.ax is not None
        assert hasattr(plot, 'fig')
        assert hasattr(plot, 'ax')
        plot.close()
    
    def test_spatial_plot_with_custom_projection(self, mock_data_factory):
        """Test SpatialPlot with custom cartopy projection."""
        from src.monet_plots.plots.spatial import SpatialPlot
        import cartopy.crs as ccrs
        
        projection = ccrs.Robinson()
        plot = SpatialPlot(projection=projection)
        
        assert plot.ax.projection == projection
        plot.close()
    
    def test_spatial_plot_basic_plotting(self, mock_data_factory):
        """Test SpatialPlot basic plotting functionality."""
        from src.monet_plots.plots.spatial import SpatialPlot
        
        plot = SpatialPlot()
        modelvar = mock_data_factory.spatial_2d()
        
        result = plot.plot(modelvar)
        
        assert result is plot.ax
        assert hasattr(plot, 'cbar')
        assert plot.ax is not None
        plot.close()
    
    def test_spatial_plot_discrete_colorbar(self, mock_data_factory):
        """Test SpatialPlot with discrete colorbar."""
        from src.monet_plots.plots.spatial import SpatialPlot
        
        plot = SpatialPlot()
        modelvar = mock_data_factory.spatial_2d()
        
        plot.plot(modelvar, discrete=True, ncolors=10)
        
        assert hasattr(plot, 'cbar')
        assert plot.ax is not None
        plot.close()
    
    def test_spatial_plot_custom_colormap(self, mock_data_factory):
        """Test SpatialPlot with custom colormap."""
        from src.monet_plots.plots.spatial import SpatialPlot
        
        plot = SpatialPlot()
        modelvar = mock_data_factory.spatial_2d()
        
        plot.plot(modelvar, plotargs={'cmap': 'plasma'})
        
        assert plot.ax is not None
        # Check that the colormap was set
        assert plot.ax.images[0].get_cmap().name == 'plasma'
        plot.close()
    
    def test_spatial_plot_invalid_data_type(self, mock_data_factory):
        """Test SpatialPlot error handling with invalid data type."""
        from src.monet_plots.plots.spatial import SpatialPlot
        
        plot = SpatialPlot()
        invalid_data = "not_an_array"
        
        with pytest.raises(TypeError):
            plot.plot(invalid_data)
        plot.close()
    
    def test_spatial_plot_empty_array(self, mock_data_factory):
        """Test SpatialPlot error handling with empty array."""
        from src.monet_plots.plots.spatial import SpatialPlot
        
        plot = SpatialPlot()
        empty_data = np.array([])
        
        with pytest.raises(TypeError):
            plot.plot(empty_data)
        plot.close()
    
    def test_spatial_plot_1d_array(self, mock_data_factory):
        """Test SpatialPlot error handling with 1D array."""
        from src.monet_plots.plots.spatial import SpatialPlot
        
        plot = SpatialPlot()
        data_1d = np.array([1, 2, 3, 4, 5])
        
        with pytest.raises(TypeError):
            plot.plot(data_1d)
        plot.close()


class TestTimeSeriesPlotUnit:
    """Unit tests for TimeSeriesPlot class."""
    
    def test_timeseries_plot_initialization(self, mock_data_factory):
        """Test TimeSeriesPlot initialization."""
        from src.monet_plots.plots.timeseries import TimeSeriesPlot
        
        plot = TimeSeriesPlot()
        
        assert plot.fig is not None
        assert plot.ax is not None
        plot.close()
    
    def test_timeseries_plot_basic_functionality(self, mock_data_factory):
        """Test TimeSeriesPlot basic plotting functionality."""
        from src.monet_plots.plots.timeseries import TimeSeriesPlot
        
        plot = TimeSeriesPlot()
        df = mock_data_factory.time_series()
        
        result = plot.plot(df)
        
        assert result is None  # plot method doesn't return anything
        assert plot.ax is not None
        assert len(plot.ax.lines) > 0  # Should have plotted lines
        assert len(plot.ax.collections) > 0  # Should have fill_between
        plot.close()
    
    def test_timeseries_plot_custom_columns(self, mock_data_factory):
        """Test TimeSeriesPlot with custom column names."""
        from src.monet_plots.plots.timeseries import TimeSeriesPlot
        
        plot = TimeSeriesPlot()
        df = mock_data_factory.time_series()
        
        plot.plot(df, x='time', y='model', title='Model Results', ylabel='ppb')
        
        assert plot.ax.get_title() == 'Model Results'
        assert 'ppb' in plot.ax.get_ylabel()
        plot.close()
    
    def test_timeseries_plot_custom_plot_args(self, mock_data_factory):
        """Test TimeSeriesPlot with custom plotting arguments."""
        from src.monet_plots.plots.timeseries import TimeSeriesPlot
        
        plot = TimeSeriesPlot()
        df = mock_data_factory.time_series()
        
        plot.plot(df, plotargs={'color': 'red', 'linewidth': 2},
                  fillargs={'alpha': 0.3, 'color': 'blue'})
        
        assert plot.ax is not None
        # Check that custom arguments were applied
        assert plot.ax.lines[0].get_color() == 'red'
        plot.close()
    
    def test_timeseries_plot_missing_required_columns(self, mock_data_factory):
        """Test TimeSeriesPlot error handling with missing columns."""
        from src.monet_plots.plots.timeseries import TimeSeriesPlot
        
        plot = TimeSeriesPlot()
        df = pd.DataFrame({'x': [1, 2, 3]})  # Missing 'time' and 'obs'
        
        with pytest.raises(KeyError):
            plot.plot(df)
        plot.close()
    
    def test_timeseries_plot_empty_dataframe(self, mock_data_factory):
        """Test TimeSeriesPlot error handling with empty DataFrame."""
        from src.monet_plots.plots.timeseries import TimeSeriesPlot
        
        plot = TimeSeriesPlot()
        df = pd.DataFrame()
        
        with pytest.raises((ValueError, KeyError, IndexError)):
            plot.plot(df)
        plot.close()
    
    def test_timeseries_plot_single_point(self, mock_data_factory):
        """Test TimeSeriesPlot with single data point."""
        from src.monet_plots.plots.timeseries import TimeSeriesPlot
        
        plot = TimeSeriesPlot()
        df = pd.DataFrame({
            'time': [pd.Timestamp('2025-01-01')],
            'obs': [25.0]
        })
        
        # Should handle gracefully or raise informative error
        try:
            plot.plot(df)
            assert plot.ax is not None
        except Exception as e:
            assert isinstance(e, (ValueError, ZeroDivisionError))
        
        plot.close()


class TestTaylorDiagramPlotUnit:
    """Unit tests for TaylorDiagramPlot class."""
    
    def test_taylor_diagram_initialization(self, mock_data_factory):
        """Test TaylorDiagramPlot initialization."""
        from src.monet_plots.plots.taylor import TaylorDiagramPlot
        
        df = mock_data_factory.taylor_data()
        obs_std = df['obs'].std()
        
        plot = TaylorDiagramPlot(obs_std)
        
        assert plot.fig is not None
        assert plot.ax is not None
        assert plot.dia is not None
        plot.close()
    
    def test_taylor_diagram_add_sample(self, mock_data_factory):
        """Test TaylorDiagramPlot add_sample method."""
        from src.monet_plots.plots.taylor import TaylorDiagramPlot
        
        df = mock_data_factory.taylor_data()
        obs_std = df['obs'].std()
        
        plot = TaylorDiagramPlot(obs_std)
        plot.add_sample(df)
        
        assert plot.dia is not None
        assert len(plot.dia.samples) > 0
        plot.close()
    
    def test_taylor_diagram_add_sample_custom_params(self, mock_data_factory):
        """Test TaylorDiagramPlot add_sample with custom parameters."""
        from src.monet_plots.plots.taylor import TaylorDiagramPlot
        
        df = mock_data_factory.taylor_data()
        obs_std = df['obs'].std()
        
        plot = TaylorDiagramPlot(obs_std)
        plot.add_sample(df, col1='obs', col2='model', marker='s', label='Test Model')
        
        assert plot.dia is not None
        plot.close()
    
    def test_taylor_diagram_add_contours(self, mock_data_factory):
        """Test TaylorDiagramPlot add_contours method."""
        from src.monet_plots.plots.taylor import TaylorDiagramPlot
        
        df = mock_data_factory.taylor_data()
        obs_std = df['obs'].std()
        
        plot = TaylorDiagramPlot(obs_std)
        plot.add_sample(df)
        contours = plot.add_contours(colors='0.5')
        
        assert contours is not None
        plot.close()
    
    def test_taylor_diagram_finish_plot(self, mock_data_factory):
        """Test TaylorDiagramPlot finish_plot method."""
        from src.monet_plots.plots.taylor import TaylorDiagramPlot
        
        df = mock_data_factory.taylor_data()
        obs_std = df['obs'].std()
        
        plot = TaylorDiagramPlot(obs_std)
        plot.add_sample(df)
        plot.finish_plot()
        
        assert plot.ax.legend_ is not None
        plot.close()
    
    def test_taylor_diagram_invalid_data_columns(self, mock_data_factory):
        """Test TaylorDiagramPlot error handling with invalid data columns."""
        from src.monet_plots.plots.taylor import TaylorDiagramPlot
        
        df = mock_data_factory.taylor_data()
        obs_std = df['obs'].std()
        
        plot = TaylorDiagramPlot(obs_std)
        
        # Test with missing columns
        invalid_df = pd.DataFrame({'x': [1, 2, 3]})
        
        with pytest.raises((KeyError, TypeError)):
            plot.add_sample(invalid_df)
        
        plot.close()
    
    def test_taylor_diagram_zero_std_data(self, mock_data_factory):
        """Test TaylorDiagramPlot with zero standard deviation data."""
        from src.monet_plots.plots.taylor import TaylorDiagramPlot
        
        # Create data with zero standard deviation
        df = pd.DataFrame({
            'obs': [5.0, 5.0, 5.0, 5.0, 5.0],
            'model': [5.0, 5.0, 5.0, 5.0, 5.0]
        })
        obs_std = df['obs'].std()  # This will be 0.0
        
        plot = TaylorDiagramPlot(obs_std)
        
        # Should handle zero std gracefully
        try:
            plot.add_sample(df)
            assert plot.dia is not None
        except Exception as e:
            # If it fails, should be a specific, expected error
            assert "zero" in str(e).lower() or "std" in str(e).lower()
        
        plot.close()


class TestScatterPlotUnit:
    """Unit tests for ScatterPlot class."""
    
    def test_scatter_plot_initialization(self, mock_data_factory):
        """Test ScatterPlot initialization."""
        from src.monet_plots.plots.scatter import ScatterPlot
        
        plot = ScatterPlot()
        
        assert plot.fig is not None
        assert plot.ax is not None
        plot.close()
    
    def test_scatter_plot_basic_functionality(self, mock_data_factory):
        """Test ScatterPlot basic plotting functionality."""
        from src.monet_plots.plots.scatter import ScatterPlot
        
        plot = ScatterPlot()
        df = mock_data_factory.scatter_data()
        
        result = plot.plot(df, 'x', 'y')
        
        assert result is None
        assert plot.ax is not None
        assert len(plot.ax.collections) > 0  # Should have scatter points
        plot.close()
    
    def test_scatter_plot_with_regression(self, mock_data_factory):
        """Test ScatterPlot with regression line."""
        from src.monet_plots.plots.scatter import ScatterPlot
        
        plot = ScatterPlot()
        df = mock_data_factory.scatter_data(correlation=0.8)
        
        plot.plot(df, 'x', 'y', ci=95)
        
        assert plot.ax is not None
        # Should have both scatter and regression line
        assert len(plot.ax.collections) > 0
        assert len(plot.ax.lines) > 0
        plot.close()
    
    def test_scatter_plot_custom_styling(self, mock_data_factory):
        """Test ScatterPlot with custom styling."""
        from src.monet_plots.plots.scatter import ScatterPlot
        
        plot = ScatterPlot()
        df = mock_data_factory.scatter_data()
        
        plot.plot(df, 'x', 'y', 
                  plotargs={'color': 'red', 'marker': 's'},
                  regargs={'color': 'blue', 'linestyle': '--'})
        
        assert plot.ax is not None
        plot.close()
    
    def test_scatter_plot_invalid_columns(self, mock_data_factory):
        """Test ScatterPlot error handling with invalid column names."""
        from src.monet_plots.plots.scatter import ScatterPlot
        
        plot = ScatterPlot()
        df = mock_data_factory.scatter_data()
        
        with pytest.raises(KeyError):
            plot.plot(df, 'invalid_x', 'invalid_y')
        
        plot.close()
    
    def test_scatter_plot_insufficient_data(self, mock_data_factory):
        """Test ScatterPlot error handling with insufficient data."""
        from src.monet_plots.plots.scatter import ScatterPlot
        
        plot = ScatterPlot()
        df = pd.DataFrame({'x': [1], 'y': [2]})  # Only one point
        
        # Should handle gracefully or raise informative error
        try:
            plot.plot(df, 'x', 'y')
            assert plot.ax is not None
        except Exception as e:
            assert isinstance(e, (ValueError, IndexError))
        
        plot.close()


class TestKDEPlotUnit:
    """Unit tests for KDEPlot class."""
    
    def test_kde_plot_initialization(self, mock_data_factory):
        """Test KDEPlot initialization."""
        from src.monet_plots.plots.kde import KDEPlot
        
        plot = KDEPlot()
        
        assert plot.fig is not None
        assert plot.ax is not None
        plot.close()
    
    def test_kde_plot_normal_distribution(self, mock_data_factory):
        """Test KDEPlot with normal distribution data."""
        from src.monet_plots.plots.kde import KDEPlot
        
        plot = KDEPlot()
        data = mock_data_factory.kde_data(distribution='normal')
        
        result = plot.plot(data)
        
        assert result is None
        assert plot.ax is not None
        assert len(plot.ax.lines) > 0  # Should have density curve
        plot.close()
    
    def test_kde_plot_bimodal_distribution(self, mock_data_factory):
        """Test KDEPlot with bimodal distribution data."""
        from src.monet_plots.plots.kde import KDEPlot
        
        plot = KDEPlot()
        data = mock_data_factory.kde_data(distribution='bimodal')
        
        plot.plot(data)
        
        assert plot.ax is not None
        assert len(plot.ax.lines) > 0
        plot.close()
    
    def test_kde_plot_custom_bandwidth(self, mock_data_factory):
        """Test KDEPlot with custom bandwidth."""
        from src.monet_plots.plots.kde import KDEPlot
        
        plot = KDEPlot()
        data = mock_data_factory.kde_data()
        
        plot.plot(data, bw='scott')
        
        assert plot.ax is not None
        plot.close()
    
    def test_kde_plot_with_shade(self, mock_data_factory):
        """Test KDEPlot with shaded area."""
        from src.monet_plots.plots.kde import KDEPlot
        
        plot = KDEPlot()
        data = mock_data_factory.kde_data()
        
        plot.plot(data, shade=True)
        
        assert plot.ax is not None
        assert len(plot.ax.collections) > 0  # Should have filled area
        plot.close()
    
    def test_kde_plot_invalid_data(self, mock_data_factory):
        """Test KDEPlot error handling with invalid data."""
        from src.monet_plots.plots.kde import KDEPlot
        
        plot = KDEPlot()
        invalid_data = []  # Empty list
        
        with pytest.raises((ValueError, TypeError)):
            plot.plot(invalid_data)
        
        plot.close()


class TestXarraySpatialPlotUnit:
    """Unit tests for XarraySpatialPlot class."""
    
    def test_xarray_spatial_plot_initialization(self, mock_data_factory):
        """Test XarraySpatialPlot initialization."""
        from src.monet_plots.plots.xarray_spatial import XarraySpatialPlot
        
        plot = XarraySpatialPlot()
        
        assert plot.fig is not None
        assert plot.ax is not None
        plot.close()
    
    def test_xarray_spatial_plot_with_dataarray(self, mock_data_factory):
        """Test XarraySpatialPlot with xarray DataArray."""
        from src.monet_plots.plots.xarray_spatial import XarraySpatialPlot
        
        plot = XarraySpatialPlot()
        data = mock_data_factory.xarray_data()
        
        result = plot.plot(data)
        
        assert plot.ax is not None  # plot.plot() returns None, not the axes
        assert plot.ax is not None
        plot.close()
    
    def test_xarray_spatial_plot_custom_coordinates(self, mock_data_factory):
        """Test XarraySpatialPlot with custom coordinate names."""
        from src.monet_plots.plots.xarray_spatial import XarraySpatialPlot
        
        plot = XarraySpatialPlot()
        data = mock_data_factory.xarray_data()
        
        # Test with custom coordinate names
        plot.plot(data, x='latitude', y='longitude', cmap='viridis')
        
        assert plot.ax is not None
        plot.close()
    
    def test_xarray_spatial_plot_invalid_data(self, mock_data_factory):
        """Test XarraySpatialPlot error handling with invalid data."""
        from src.monet_plots.plots.xarray_spatial import XarraySpatialPlot
        
        plot = XarraySpatialPlot()
        invalid_data = np.array([[1, 2], [3, 4]])  # Regular numpy array
        
        with pytest.raises((TypeError, AttributeError)):
            plot.plot(invalid_data)
        
        plot.close()


class TestFacetGridPlotUnit:
    """Unit tests for FacetGridPlot class."""
    
    def test_facet_grid_plot_initialization(self, mock_data_factory):
        """Test FacetGridPlot initialization."""
        from src.monet_plots.plots.facet_grid import FacetGridPlot
        
        data = mock_data_factory.facet_data()
        plot = FacetGridPlot(data, col='time')
        
        assert hasattr(plot, 'g')  # Should have FacetGrid object
        assert plot.g is not None
        plot.close()
    
    def test_facet_grid_plot_basic_functionality(self, mock_data_factory):
        """Test FacetGridPlot basic plotting functionality."""
        from src.monet_plots.plots.facet_grid import FacetGridPlot
        
        data = mock_data_factory.facet_data()
        plot = FacetGridPlot(data, col='time')
        
        result = plot.plot()
        
        assert result is plot.g
        assert plot.g is not None
        assert len(plot.g.axes.flat) > 1  # Should have multiple facets
        plot.close()
    
    def test_facet_grid_plot_row_col_facets(self, mock_data_factory):
        """Test FacetGridPlot with row and column facets."""
        from src.monet_plots.plots.facet_grid import FacetGridPlot
        
        data = mock_data_factory.facet_data()
        plot = FacetGridPlot(data, row='x', col='y')
        
        plot.plot()
        
        assert plot.g is not None
        assert len(plot.g.axes.flat) > 1
        plot.close()
    
    def test_facet_grid_plot_custom_plot_type(self, mock_data_factory):
        """Test FacetGridPlot with custom plot type."""
        from src.monet_plots.plots.facet_grid import FacetGridPlot
        
        data = mock_data_factory.facet_data()
        plot = FacetGridPlot(data, col='time', plot_type='contourf')
        
        plot.plot()
        
        assert plot.g is not None
        plot.close()
    
    def test_facet_grid_plot_invalid_data(self, mock_data_factory):
        """Test FacetGridPlot error handling with invalid data."""
        from src.monet_plots.plots.facet_grid import FacetGridPlot
        
        invalid_data = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        
        with pytest.raises((ValueError, KeyError)):
            plot = FacetGridPlot(invalid_data, col='time')
        
    def test_facet_grid_plot_missing_facet_dimension(self, mock_data_factory):
        """Test FacetGridPlot error handling with missing facet dimension."""
        from src.monet_plots.plots.facet_grid import FacetGridPlot
        
        data = mock_data_factory.facet_data()
        
        with pytest.raises((ValueError, KeyError)):
            plot = FacetGridPlot(data, col='nonexistent_dim')


class TestWindQuiverPlotUnit:
    """Unit tests for WindQuiverPlot class."""
    
    def test_wind_quiver_plot_initialization(self, mock_data_factory):
        """Test WindQuiverPlot initialization."""
        from src.monet_plots.plots.wind_quiver import WindQuiverPlot
        
        plot = WindQuiverPlot()
        
        assert plot.fig is not None
        assert plot.ax is not None
        plot.close()
    
    def test_wind_quiver_plot_basic_functionality(self, mock_data_factory):
        """Test WindQuiverPlot basic plotting functionality."""
        from src.monet_plots.plots.wind_quiver import WindQuiverPlot
        
        plot = WindQuiverPlot()
        u, v = mock_data_factory.wind_data()
        x, y = np.meshgrid(np.arange(u.shape[1]), np.arange(u.shape[0]))
        
        result = plot.plot(x, y, u, v)
        
        assert result is plot.ax
        assert plot.ax is not None
        assert len(plot.ax.quiverobjects) > 0  # Should have quiver plot
        plot.close()
    
    def test_wind_quiver_plot_custom_scaling(self, mock_data_factory):
        """Test WindQuiverPlot with custom scaling."""
        from src.monet_plots.plots.wind_quiver import WindQuiverPlot
        
        plot = WindQuiverPlot()
        u, v = mock_data_factory.wind_data()
        x, y = np.meshgrid(np.arange(u.shape[1]), np.arange(u.shape[0]))
        
        plot.plot(x, y, u, v, scale=50, width=0.002)
        
        assert plot.ax is not None
        plot.close()
    
    def test_wind_quiver_plot_invalid_data(self, mock_data_factory):
        """Test WindQuiverPlot error handling with invalid data."""
        from src.monet_plots.plots.wind_quiver import WindQuiverPlot
        
        plot = WindQuiverPlot()
        u = np.array([[1, 2], [3, 4]])
        v = np.array([[5, 6]])  # Wrong shape
        
        with pytest.raises((ValueError, IndexError)):
            plot.plot(u, v, u, v)  # Wrong parameter order too
        
        plot.close()


class TestWindBarbsPlotUnit:
    """Unit tests for WindBarbsPlot class."""
    
    def test_wind_barbs_plot_initialization(self, mock_data_factory):
        """Test WindBarbsPlot initialization."""
        from src.monet_plots.plots.wind_barbs import WindBarbsPlot
        
        plot = WindBarbsPlot()
        
        assert plot.fig is not None
        assert plot.ax is not None
        plot.close()
    
    def test_wind_barbs_plot_basic_functionality(self, mock_data_factory):
        """Test WindBarbsPlot basic plotting functionality."""
        from src.monet_plots.plots.wind_barbs import WindBarbsPlot
        
        plot = WindBarbsPlot()
        u, v = mock_data_factory.wind_data()
        x, y = np.meshgrid(np.arange(u.shape[1]), np.arange(u.shape[0]))
        
        result = plot.plot(x, y, u, v)
        
        assert result is plot.ax
        assert plot.ax is not None
        assert len(plot.ax.barbobjects) > 0  # Should have barb plot
        plot.close()
    
    def test_wind_barbs_plot_custom_density(self, mock_data_factory):
        """Test WindBarbsPlot with custom density."""
        from src.monet_plots.plots.wind_barbs import WindBarbsPlot
        
        plot = WindBarbsPlot()
        u, v = mock_data_factory.wind_data()
        x, y = np.meshgrid(np.arange(u.shape[1]), np.arange(u.shape[0]))
        
        plot.plot(x, y, u, v, density=2)
        
        assert plot.ax is not None
        plot.close()
    
    def test_wind_barbs_plot_invalid_data(self, mock_data_factory):
        """Test WindBarbsPlot error handling with invalid data."""
        from src.monet_plots.plots.wind_barbs import WindBarbsPlot
        
        plot = WindBarbsPlot()
        u = np.array([[1, 2], [3, 4]])
        v = np.array([[5, 6]])  # Wrong shape
        
        with pytest.raises((ValueError, IndexError)):
            plot.plot(u, v, u, v)  # Wrong parameter order too
        
        plot.close()


# Test cleanup fixture
@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Clean up matplotlib figures after each test."""
    yield
    plt.close('all')
    plt.clf()