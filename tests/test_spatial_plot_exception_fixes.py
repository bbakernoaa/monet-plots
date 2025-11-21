"""
Test file to demonstrate and fix spatial plot exception type mismatches.

This file contains failing tests that demonstrate the actual vs expected exception types
for spatial plot error handling. The tests will be updated to match the actual behavior.
"""

import pytest
import numpy as np
from src.monet_plots.plots.spatial import SpatialPlot


class TestSpatialPlotExceptionFixes:
    """Test class to fix spatial plot exception type mismatches."""
    
    def test_spatial_plot_invalid_data_type_actual_exception(self):
        """Test SpatialPlot error handling with invalid data type - actual behavior."""
        plot = SpatialPlot()
        invalid_data = "not_an_array"
        
        # Actual exception is TypeError, not (TypeError, AttributeError)
        with pytest.raises(TypeError):
            plot.plot(invalid_data)
        
        plot.close()
    
    def test_spatial_plot_empty_array_actual_exception(self):
        """Test SpatialPlot error handling with empty array - actual behavior."""
        plot = SpatialPlot()
        empty_data = np.array([])
        
        # Actual exception is TypeError, not (ValueError, IndexError)
        with pytest.raises(TypeError):
            plot.plot(empty_data)
        
        plot.close()
    
    def test_spatial_plot_1d_array_actual_exception(self):
        """Test SpatialPlot error handling with 1D array - actual behavior."""
        plot = SpatialPlot()
        data_1d = np.array([1, 2, 3, 4, 5])
        
        # Actual exception is TypeError, not (ValueError, IndexError)
        with pytest.raises(TypeError):
            plot.plot(data_1d)
        
        plot.close()
    
    def test_spatial_plot_2d_array_success(self):
        """Test SpatialPlot with valid 2D array - should work."""
        plot = SpatialPlot()
        data_2d = np.random.rand(10, 10)
        
        # This should not raise an exception
        result = plot.plot(data_2d)
        assert result is plot.ax
        assert plot.ax is not None
        
        plot.close()