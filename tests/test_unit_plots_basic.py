"""
Basic Unit Tests for MONET Plots - Individual Plot Classes

This module contains unit tests for basic plot classes using TDD approach.
"""

import pytest
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os


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


# Test cleanup fixture
@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Clean up matplotlib figures after each test."""
    yield
    plt.close('all')
    plt.clf()