"""
Test file to add parameter validation to plot methods.

This file contains tests that demonstrate the need for parameter validation
and the implementation of validation functions for plot methods.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
from src.monet_plots.plots.spatial import SpatialPlot
from src.monet_plots.plots.timeseries import TimeSeriesPlot


class TestParameterValidationAdditions:
    """Test class to add parameter validation to plot methods."""
    
    def test_spatial_plot_parameter_validation_invalid_discrete(self, mock_data_factory):
        """Test SpatialPlot parameter validation for invalid discrete parameter."""
        plot = SpatialPlot()
        modelvar = mock_data_factory.spatial_2d()
        
        # Test invalid discrete parameter type
        with pytest.raises((TypeError, ValueError)):
            plot.plot(modelvar, discrete="invalid_boolean")
        
        # Test invalid ncolors parameter
        with pytest.raises((TypeError, ValueError)):
            plot.plot(modelvar, discrete=True, ncolors="invalid")
        
        plot.close()
    
    def test_spatial_plot_parameter_validation_invalid_plotargs(self, mock_data_factory):
        """Test SpatialPlot parameter validation for invalid plotargs."""
        plot = SpatialPlot()
        modelvar = mock_data_factory.spatial_2d()
        
        # Test invalid plotargs type
        with pytest.raises(TypeError):
            plot.plot(modelvar, plotargs="invalid_dict")
        
        # Test invalid colormap name
        with pytest.raises((ValueError, KeyError)):
            plot.plot(modelvar, plotargs={'cmap': 'nonexistent_colormap'})
        
        plot.close()
    
    def test_timeseries_plot_parameter_validation_invalid_columns(self, mock_data_factory):
        """Test TimeSeriesPlot parameter validation for invalid column names."""
        plot = TimeSeriesPlot()
        df = mock_data_factory.time_series()
        
        # Test invalid column names
        with pytest.raises((KeyError, ValueError)):
            plot.plot(df, x='invalid_column', y='model')
        
        with pytest.raises((KeyError, ValueError)):
            plot.plot(df, x='time', y='invalid_column')
        
        plot.close()
    
    def test_timeseries_plot_parameter_validation_invalid_plotargs(self, mock_data_factory):
        """Test TimeSeriesPlot parameter validation for invalid plot arguments."""
        plot = TimeSeriesPlot()
        df = mock_data_factory.time_series()
        
        # Test invalid plotargs type
        with pytest.raises(TypeError):
            plot.plot(df, plotargs="invalid")
        
        # Test invalid fillargs type
        with pytest.raises(TypeError):
            plot.plot(df, fillargs="invalid")
        
        plot.close()
    
    def test_spatial_plot_parameter_validation_edge_cases(self, mock_data_factory):
        """Test SpatialPlot parameter validation edge cases."""
        plot = SpatialPlot()
        modelvar = mock_data_factory.spatial_2d()
        
        # Test very large ncolors
        with pytest.raises((ValueError, OverflowError)):
            plot.plot(modelvar, discrete=True, ncolors=999999)
        
        # Test negative ncolors
        with pytest.raises(ValueError):
            plot.plot(modelvar, discrete=True, ncolors=-10)
        
        plot.close()
    
    def test_timeseries_plot_parameter_validation_edge_cases(self, mock_data_factory):
        """Test TimeSeriesPlot parameter validation edge cases."""
        plot = TimeSeriesPlot()
        df = mock_data_factory.time_series()
        
        # Test invalid linewidth
        with pytest.raises((TypeError, ValueError)):
            plot.plot(df, plotargs={'linewidth': 'invalid'})
        
        # Test invalid alpha
        with pytest.raises((TypeError, ValueError)):
            plot.plot(df, fillargs={'alpha': 1.5})  # alpha > 1.0
        
        plot.close()


def validate_plot_parameters(plot_class, method, **kwargs):
    """
    Validate parameters for plot methods.
    
    Args:
        plot_class: The plot class name
        method: The method name
        **kwargs: Parameters to validate
        
    Raises:
        TypeError: If parameter types are invalid
        ValueError: If parameter values are invalid
    """
    if plot_class == 'SpatialPlot' and method == 'plot':
        # Validate discrete parameter
        if 'discrete' in kwargs:
            discrete = kwargs['discrete']
            if not isinstance(discrete, bool):
                raise TypeError(f"discrete parameter must be boolean, got {type(discrete).__name__}")
        
        # Validate ncolors parameter
        if 'ncolors' in kwargs:
            ncolors = kwargs['ncolors']
            if not isinstance(ncolors, int):
                raise TypeError(f"ncolors parameter must be integer, got {type(ncolors).__name__}")
            if ncolors <= 0 or ncolors > 1000:
                raise ValueError(f"ncolors parameter must be between 1 and 1000, got {ncolors}")
        
        # Validate plotargs parameter
        if 'plotargs' in kwargs and kwargs['plotargs'] is not None:
            plotargs = kwargs['plotargs']
            if not isinstance(plotargs, dict):
                raise TypeError(f"plotargs parameter must be dict, got {type(plotargs).__name__}")
            
            # Validate specific plotargs keys
            if 'cmap' in plotargs:
                cmap = plotargs['cmap']
                # This would need actual colormap validation
                if not isinstance(cmap, str):
                    raise TypeError(f"colormap must be string, got {type(cmap).__name__}")
    
    elif plot_class == 'TimeSeriesPlot' and method == 'plot':
        # Validate x parameter
        if 'x' in kwargs:
            x = kwargs['x']
            if not isinstance(x, str):
                raise TypeError(f"x parameter must be string, got {type(x).__name__}")
        
        # Validate y parameter  
        if 'y' in kwargs:
            y = kwargs['y']
            if not isinstance(y, str):
                raise TypeError(f"y parameter must be string, got {type(y).__name__}")
        
        # Validate plotargs parameter
        if 'plotargs' in kwargs and kwargs['plotargs'] is not None:
            plotargs = kwargs['plotargs']
            if not isinstance(plotargs, dict):
                raise TypeError(f"plotargs parameter must be dict, got {type(plotargs).__name__}")
        
        # Validate fillargs parameter
        if 'fillargs' in kwargs and kwargs['fillargs'] is not None:
            fillargs = kwargs['fillargs']
            if not isinstance(fillargs, dict):
                raise TypeError(f"fillargs parameter must be dict, got {type(fillargs).__name__}")
            
            # Validate alpha in fillargs
            if 'alpha' in fillargs:
                alpha = fillargs['alpha']
                if not isinstance(alpha, (int, float)):
                    raise TypeError(f"alpha must be numeric, got {type(alpha).__name__}")
                if not 0 <= alpha <= 1:
                    raise ValueError(f"alpha must be between 0 and 1, got {alpha}")


# Enhanced plot methods with parameter validation
def enhanced_spatial_plot_with_validation(plot_instance, data, **kwargs):
    """
    Enhanced SpatialPlot.plot method with parameter validation.
    """
    validate_plot_parameters('SpatialPlot', 'plot', **kwargs)
    return plot_instance.plot(data, **kwargs)


def enhanced_timeseries_plot_with_validation(plot_instance, data, **kwargs):
    """
    Enhanced TimeSeriesPlot.plot method with parameter validation.
    """
    validate_plot_parameters('TimeSeriesPlot', 'plot', **kwargs)
    return plot_instance.plot(data, **kwargs)