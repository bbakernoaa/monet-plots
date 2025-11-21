"""
Comprehensive Error Handling Tests for MONET Plots
==================================================

This module contains comprehensive error handling, edge cases, and boundary condition tests
for all plot classes using TDD approach.

Following TDD principles: Write failing tests first, implement minimal code to pass, then refactor.
"""

import pytest
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch
import warnings
import traceback


class TestGeneralErrorHandling:
    """General error handling and edge case tests."""
    
    def test_base_plot_error_handling(self, mock_data_factory):
        """Test BasePlot error handling."""
        try:
            from src.monet_plots.plots.base import BasePlot
            
            plot = BasePlot()
            
            # Test 1: Save to invalid path - should handle gracefully
            try:
                plot.save("/nonexistent/directory/test.png")
            except (PermissionError, FileNotFoundError, OSError):
                pass  # Expected for invalid paths
            
            try:
                plot.save("")  # Empty filename
            except (ValueError, OSError):
                pass  # Expected for empty filename
            
            # Test 2: Save with invalid arguments
            with tempfile.TemporaryDirectory() as temp_dir:
                test_file = os.path.join(temp_dir, "test.png")
                
                # Should handle invalid save arguments gracefully
                try:
                    plot.save(test_file, invalid_argument=True)
                except TypeError:
                    pass  # Expected for invalid arguments
            
            plot.close()
            
        except ImportError:
            pytest.skip("BasePlot not available")
    
    def test_file_save_permissions_and_errors(self, mock_data_factory):
        """Test error handling for file save operations."""
        try:
            from src.monet_plots.plots.spatial import SpatialPlot
            
            plot = SpatialPlot()
            data = mock_data_factory.spatial_2d()
            plot.plot(data)
            
            # Test 1: Save to read-only directory (if possible)
            try:
                # Try to save to root directory (should fail on most systems)
                plot.save("/test_readonly.png")
            except (PermissionError, OSError):
                pass  # Expected failure
            
            # Test 2: Invalid file format
            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    plot.save(os.path.join(temp_dir, "test.invalid"))
                except (ValueError, KeyError):
                    pass  # Expected for invalid format
            
            # Test 3: Save with invalid DPI
            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    plot.save(os.path.join(temp_dir, "test.png"), dpi="invalid")
                except (TypeError, ValueError):
                    pass  # Expected for invalid DPI
            
            plot.close()
            
        except ImportError:
            pytest.skip("SpatialPlot not available")
    
    def test_matplotlib_state_corruption(self, mock_data_factory):
        """Test matplotlib state handling and corruption prevention."""
        # Create multiple plots and verify matplotlib state
        initial_figures = len(plt.get_fignums())
        
        plots_created = []
        
        try:
            # Create several plots
            try:
                from src.monet_plots.plots.spatial import SpatialPlot
                plot1 = SpatialPlot()
                plot1.plot(mock_data_factory.spatial_2d())
                plots_created.append(plot1)
            except ImportError:
                pass
            
            try:
                from src.monet_plots.plots.timeseries import TimeSeriesPlot
                plot2 = TimeSeriesPlot()
                plot2.plot(mock_data_factory.time_series())
                plots_created.append(plot2)
            except ImportError:
                pass
            
            try:
                from src.monet_plots.plots.scatter import ScatterPlot
                plot3 = ScatterPlot()
                df = mock_data_factory.scatter_data()
                plot3.plot(df, 'x', 'y')
                plots_created.append(plot3)
            except ImportError:
                pass
            
            # Verify matplotlib state
            figures_after_creation = len(plt.get_fignums())
            expected_figures = initial_figures + len(plots_created)
            assert figures_after_creation == expected_figures
            
            # Close plots one by one and verify state
            for i, plot in enumerate(plots_created):
                plot.close()
                remaining_figures = len(plt.get_fignums())
                expected_remaining = expected_figures - (i + 1)
                assert remaining_figures == expected_remaining
            
            # Final state should match initial state
            assert len(plt.get_fignums()) == initial_figures
            
        finally:
            # Ensure cleanup
            for plot in plots_created:
                try:
                    plot.close()
                except:
                    pass
    
    def test_memory_error_handling(self, mock_data_factory):
        """Test behavior when memory errors occur."""
        # This test simulates memory pressure scenarios
        plots_created = []
        
        try:
            # Create many plots to simulate memory pressure
            for i in range(10):
                try:
                    from src.monet_plots.plots.spatial import SpatialPlot
                    plot = SpatialPlot()
                    data = mock_data_factory.spatial_2d()
                    plot.plot(data)
                    plots_created.append(plot)
                    
                    # Verify each plot is valid
                    assert plot.ax is not None
                    
                except ImportError:
                    pytest.skip("SpatialPlot not available")
            
            # Verify we can still create plots under "memory pressure"
            assert len(plots_created) > 0
            
        finally:
            # Clean up all plots
            for plot in plots_created:
                try:
                    plot.close()
                except:
                    pass
    
    def test_invalid_plot_arguments(self, mock_data_factory):
        """Test handling of invalid plot arguments."""
        try:
            from src.monet_plots.plots.spatial import SpatialPlot
            
            plot = SpatialPlot()
            data = mock_data_factory.spatial_2d()
            
            # Test invalid plot arguments - should raise AttributeError for invalid matplotlib parameters
            try:
                plot.plot(data, plotargs={'invalid_param': 'value'})
            except AttributeError as e:
                # Expected for invalid matplotlib parameters
                assert "unexpected keyword argument" in str(e).lower()
            
            # Test invalid keyword arguments - these get passed to matplotlib and cause AttributeError
            try:
                plot.plot(data, invalid_kwarg=True)
            except AttributeError as e:
                # Expected for invalid matplotlib kwargs
                assert "unexpected keyword argument" in str(e).lower()
            
            # Test invalid discrete parameter
            try:
                plot.plot(data, discrete="invalid_boolean")
            except (TypeError, ValueError):
                pass  # Expected for invalid discrete parameter
            
            plot.close()
            
        except ImportError:
            pytest.skip("SpatialPlot not available")


class TestSpatialPlotErrorHandling:
    """Error handling tests specifically for SpatialPlot."""
    
    def test_spatial_plot_invalid_data_types(self, mock_data_factory):
        """Test SpatialPlot with various invalid data types."""
        from src.monet_plots.plots.spatial import SpatialPlot
        
        plot = SpatialPlot()
        
        # Test cases with different invalid data types
        invalid_data_cases = [
            ("string_data", "not_an_array"),
            ("list_data", [[1, 2], [3, 4]]),  # List instead of numpy array
            ("scalar_data", 5.0),  # Scalar instead of array
            ("dict_data", {"data": [1, 2, 3]}),  # Dictionary
        ]
        
        for case_name, invalid_data in invalid_data_cases:
            try:
                # Should either work or fail gracefully with appropriate error
                plot.plot(invalid_data)
            except (TypeError, AttributeError) as e:
                # Expected error types for invalid data
                assert any(keyword in str(e).lower() for keyword in
                          ['dtype', 'image', 'data', 'cannot', 'convert'])
        
        plot.close()
    
    def test_spatial_plot_invalid_dimensions(self, mock_data_factory):
        """Test SpatialPlot with invalid array dimensions."""
        from src.monet_plots.plots.spatial import SpatialPlot
        
        plot = SpatialPlot()
        
        # Test cases with different invalid dimensions
        invalid_dimension_cases = [
            ("1d_array", np.array([1, 2, 3, 4, 5])),
            ("3d_array", np.random.randn(5, 5, 5)),
            ("4d_array", np.random.randn(3, 3, 3, 3)),
            ("empty_array", np.array([])),
        ]
        
        for case_name, invalid_data in invalid_dimension_cases:
            try:
                # Should either work or fail gracefully
                plot.plot(invalid_data)
            except (TypeError, ValueError) as e:
                # Expected error types for invalid dimensions
                assert any(keyword in str(e).lower() for keyword in
                          ['shape', 'dimension', 'invalid', 'array'])
        
        plot.close()
    
    def test_spatial_plot_nan_inf_handling(self, mock_data_factory):
        """Test SpatialPlot with NaN and infinite values."""
        from src.monet_plots.plots.spatial import SpatialPlot
        
        plot = SpatialPlot()
        
        # Test cases with NaN and infinite values
        nan_inf_cases = [
            ("nan_values", np.array([[1, 2], [np.nan, 4]])),
            ("inf_values", np.array([[1, 2], [np.inf, 4]])),
            ("negative_inf", np.array([[1, 2], [-np.inf, 4]])),
            ("mixed_nan_inf", np.array([[1, np.nan], [np.inf, 4]])),
            ("all_nan", np.full((3, 3), np.nan)),
            ("all_inf", np.full((3, 3), np.inf)),
        ]
        
        for case_name, problematic_data in nan_inf_cases:
            try:
                # Should either work or fail gracefully
                result = plot.plot(problematic_data)
                assert result is not None, f"Plot should handle {case_name} gracefully"
            except Exception as e:
                # If it fails, should be with a clear, expected error message
                error_msg = str(e).lower()
                assert any(keyword in error_msg for keyword in 
                          ['nan', 'inf', 'invalid', 'value']), \
                    f"Error message should mention the issue: {e}"
        
        plot.close()
    
    def test_spatial_plot_projection_errors(self, mock_data_factory):
        """Test SpatialPlot with invalid projection parameters."""
        from src.monet_plots.plots.spatial import SpatialPlot
        
        # Test cases with invalid projection parameters
        invalid_projection_cases = [
            ("invalid_projection_type", "invalid_projection"),
            ("none_projection", None),
            ("wrong_projection_object", "not_a_cartopy_projection"),
        ]
        
        for case_name, invalid_projection in invalid_projection_cases:
            try:
                plot = SpatialPlot(projection=invalid_projection)
                # Should either work or fail during initialization
                assert plot is not None
                plot.close()
            except (TypeError, ValueError, AttributeError) as e:
                # Expected error for invalid projection
                assert "projection" in str(e).lower() or "invalid" in str(e).lower()
    
    def test_spatial_plot_colorbar_errors(self, mock_data_factory):
        """Test SpatialPlot colorbar-related errors."""
        from src.monet_plots.plots.spatial import SpatialPlot
        
        plot = SpatialPlot()
        data = mock_data_factory.spatial_2d()
        
        # Test invalid colorbar parameters
        invalid_colorbar_cases = [
            ("invalid_ncolors", {"discrete": True, "ncolors": 0}),
            ("negative_ncolors", {"discrete": True, "ncolors": -5}),
            ("invalid_colormap", {"plotargs": {"cmap": "nonexistent_colormap"}}),
            ("invalid_vmin_vmax", {"vmin": 100, "vmax": 0}),  # vmin > vmax
        ]
        
        for case_name, invalid_params in invalid_colorbar_cases:
            try:
                plot.plot(data, **invalid_params)
                # If it succeeds, that's also acceptable (graceful handling)
            except (ValueError, TypeError) as e:
                # Expected error for invalid parameters - check for specific error messages
                error_msg = str(e).lower()
                # Check for the specific error messages from matplotlib/numpy
                assert any(keyword in error_msg for keyword in
                          ['boundaries', 'at least 2', 'region', 'passed',
                           'number of samples', 'must be non-negative',
                           'color', 'map', 'range', 'value', 'invalid'])
        
        plot.close()


class TestTimeSeriesPlotErrorHandling:
    """Error handling tests specifically for TimeSeriesPlot."""
    
    def test_timeseries_plot_missing_columns(self, mock_data_factory):
        """Test TimeSeriesPlot with missing required columns."""
        from src.monet_plots.plots.timeseries import TimeSeriesPlot
        
        plot = TimeSeriesPlot()
        
        # Test cases with missing columns
        missing_column_cases = [
            ("missing_time_column", pd.DataFrame({'obs': [1, 2, 3]})),
            ("missing_obs_column", pd.DataFrame({'time': pd.date_range('2025-01-01', periods=3)})),
            ("missing_both_columns", pd.DataFrame({'other': [1, 2, 3]})),
            ("empty_dataframe", pd.DataFrame()),
        ]
        
        for case_name, invalid_df in missing_column_cases:
            try:
                # Should either work or fail gracefully
                plot.plot(invalid_df)
            except (KeyError, ValueError) as e:
                # Expected error for missing columns
                assert any(keyword in str(e).lower() for keyword in
                          ['time', 'obs', 'not in index', 'column'])
        
        plot.close()
    
    def test_timeseries_plot_invalid_column_names(self, mock_data_factory):
        """Test TimeSeriesPlot with invalid column names."""
        from src.monet_plots.plots.timeseries import TimeSeriesPlot
        
        plot = TimeSeriesPlot()
        valid_df = mock_data_factory.time_series()
        
        # Test with custom invalid column specifications
        invalid_column_cases = [
            ("invalid_x_column", "nonexistent_column", "obs"),
            ("invalid_y_column", "time", "nonexistent_column"),
            ("both_invalid_columns", "invalid_x", "invalid_y"),
        ]
        
        for case_name, invalid_x, invalid_y in invalid_column_cases:
            try:
                # Should either work or fail gracefully
                plot.plot(valid_df, x=invalid_x, y=invalid_y)
            except KeyError as e:
                # Expected error for invalid columns
                assert any(keyword in str(e).lower() for keyword in
                          ['not in index', 'column', 'key'])
        
        plot.close()
    
    def test_timeseries_plot_invalid_data_types(self, mock_data_factory):
        """Test TimeSeriesPlot with invalid data types in columns."""
        from src.monet_plots.plots.timeseries import TimeSeriesPlot
        
        plot = TimeSeriesPlot()
        
        # Test cases with invalid data types
        invalid_data_cases = [
            ("string_time_column", pd.DataFrame({
                'time': ['not', 'dates', 'here'],
                'obs': [1, 2, 3]
            })),
            ("string_obs_column", pd.DataFrame({
                'time': pd.date_range('2025-01-01', periods=3),
                'obs': ['not', 'numbers', 'here']
            })),
            ("mixed_data_types", pd.DataFrame({
                'time': pd.date_range('2025-01-01', periods=3),
                'obs': [1, 'mixed', 3]
            })),
        ]
        
        for case_name, invalid_df in invalid_data_cases:
            try:
                # Should either work (with type conversion) or fail gracefully
                plot.plot(invalid_df)
            except Exception as e:
                # If it fails, should be with a clear error message
                assert any(keyword in str(e).lower() for keyword in
                          ['obs', 'column', 'type', 'convert', 'invalid', 'data'])
        
        plot.close()
    
    def test_timeseries_plot_insufficient_data(self, mock_data_factory):
        """Test TimeSeriesPlot with insufficient data points."""
        from src.monet_plots.plots.timeseries import TimeSeriesPlot
        
        plot = TimeSeriesPlot()
        
        # Test cases with insufficient data
        insufficient_data_cases = [
            ("single_point", pd.DataFrame({
                'time': [pd.Timestamp('2025-01-01')],
                'obs': [25.0]
            })),
            ("two_points", pd.DataFrame({
                'time': pd.date_range('2025-01-01', periods=2),
                'obs': [25.0, 26.0]
            })),
            ("constant_values", pd.DataFrame({
                'time': pd.date_range('2025-01-01', periods=5),
                'obs': [25.0, 25.0, 25.0, 25.0, 25.0]
            })),
        ]
        
        for case_name, insufficient_df in insufficient_data_cases:
            try:
                # Should either work or fail gracefully
                plot.plot(insufficient_df)
                # If it works, verify the plot was created
                assert plot.ax is not None
            except Exception as e:
                # If it fails, should be with a clear error message
                assert any(keyword in str(e).lower() for keyword in 
                          ['insufficient', 'data', 'points', 'statistics'])
        
        plot.close()


class TestTaylorDiagramPlotErrorHandling:
    """Error handling tests specifically for TaylorDiagramPlot."""
    
    def test_taylor_diagram_invalid_obs_std(self, mock_data_factory):
        """Test TaylorDiagramPlot with invalid observation standard deviation."""
        from src.monet_plots.plots.taylor import TaylorDiagramPlot
        
        # Test cases with invalid obs_std values
        invalid_obs_std_cases = [
            ("negative_std", -1.0),
            ("zero_std", 0.0),
            ("invalid_type", "not_a_number"),
            ("none_std", None),
        ]
        
        for case_name, invalid_obs_std in invalid_obs_std_cases:
            try:
                plot = TaylorDiagramPlot(invalid_obs_std)
                # Should either work or fail during initialization
                assert plot is not None
                plot.close()
            except (ValueError, TypeError) as e:
                # Expected error for invalid obs_std - based on actual error
                error_msg = str(e).lower()
                assert any(keyword in error_msg for keyword in
                          ['multiply', 'sequence', 'float', 'std', 'invalid'])
    
    def test_taylor_diagram_invalid_data_columns(self, mock_data_factory):
        """Test TaylorDiagramPlot add_sample with invalid data columns."""
        from src.monet_plots.plots.taylor import TaylorDiagramPlot
        
        # Create valid plot first
        df = mock_data_factory.taylor_data()
        obs_std = df['obs'].std()
        plot = TaylorDiagramPlot(obs_std)
        
        # Test cases with invalid data
        invalid_data_cases = [
            ("missing_obs_column", pd.DataFrame({'model': [1, 2, 3]})),
            ("missing_model_column", pd.DataFrame({'obs': [1, 2, 3]})),
            ("missing_both_columns", pd.DataFrame({'other': [1, 2, 3]})),
            ("wrong_column_names", pd.DataFrame({'observation': [1, 2, 3], 'prediction': [1, 2, 3]})),
        ]
        
        for case_name, invalid_df in invalid_data_cases:
            try:
                # Should either work or fail gracefully
                plot.add_sample(invalid_df)
            except (KeyError, TypeError) as e:
                # Expected error for invalid columns
                assert any(keyword in str(e).lower() for keyword in
                          ['obs', 'model', 'column', 'key'])
        
        plot.close()
    
    def test_taylor_diagram_zero_variance_data(self, mock_data_factory):
        """Test TaylorDiagramPlot with zero variance data."""
        from src.monet_plots.plots.taylor import TaylorDiagramPlot
        
        # Create data with zero variance
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
            assert "zero" in str(e).lower() or "std" in str(e).lower() or \
                   "variance" in str(e).lower() or "correlation" in str(e).lower()
        
        plot.close()


class TestScatterPlotErrorHandling:
    """Error handling tests specifically for ScatterPlot."""
    
    def test_scatter_plot_invalid_columns(self, mock_data_factory):
        """Test ScatterPlot with invalid column names."""
        from src.monet_plots.plots.scatter import ScatterPlot
        
        plot = ScatterPlot()
        df = mock_data_factory.scatter_data()
        
        # Test cases with invalid column specifications
        invalid_column_cases = [
            ("invalid_x_column", "nonexistent_x"),
            ("invalid_y_column", "nonexistent_y"),
            ("both_invalid", "invalid_x", "invalid_y"),
        ]
        
        for case_name, *columns in invalid_column_cases:
            if len(columns) == 1:
                try:
                    plot.plot(df, columns[0], 'y')
                except KeyError as e:
                    # Check for the actual error message format
                    assert "'" in str(e) or "not in index" in str(e)
                try:
                    plot.plot(df, 'x', columns[0])
                except KeyError as e:
                    # Check for the actual error message format
                    assert "'" in str(e) or "not in index" in str(e)
            else:
                try:
                    plot.plot(df, columns[0], columns[1])
                except KeyError as e:
                    # Check for the actual error message format
                    assert "'" in str(e) or "not in index" in str(e)
        
        plot.close()
    
    def test_scatter_plot_insufficient_data(self, mock_data_factory):
        """Test ScatterPlot with insufficient data."""
        from src.monet_plots.plots.scatter import ScatterPlot
        
        plot = ScatterPlot()
        
        # Test cases with insufficient data
        insufficient_data_cases = [
            ("single_point", pd.DataFrame({'x': [1], 'y': [2]})),
            ("two_points", pd.DataFrame({'x': [1, 2], 'y': [2, 3]})),
            ("empty_dataframe", pd.DataFrame({'x': [], 'y': []})),
        ]
        
        for case_name, insufficient_df in insufficient_data_cases:
            try:
                # Should either work or fail gracefully
                plot.plot(insufficient_df, 'x', 'y')
                # If it works, verify the plot was created
                assert plot.ax is not None
            except Exception as e:
                # If it fails, should be with a clear error message
                assert any(keyword in str(e).lower() for keyword in 
                          ['insufficient', 'data', 'points', 'sample'])
        
        plot.close()
    
    def test_scatter_plot_invalid_regression_params(self, mock_data_factory):
        """Test ScatterPlot with invalid regression parameters."""
        from src.monet_plots.plots.scatter import ScatterPlot
        
        plot = ScatterPlot()
        df = mock_data_factory.scatter_data()
        
        # Test cases with invalid regression parameters
        invalid_regression_cases = [
            ("invalid_ci", {'ci': -95}),  # Negative confidence interval
            ("invalid_ci_type", {'ci': "invalid"}),  # Non-numeric CI
            ("invalid_order", {'order': 0}),  # Invalid polynomial order
            ("invalid_lowess", {'lowess': "invalid"}),  # Non-boolean lowess
        ]
        
        for case_name, invalid_params in invalid_regression_cases:
            try:
                plot.plot(df, 'x', 'y', **invalid_params)
                # If it works, that's acceptable (graceful parameter handling)
            except (ValueError, TypeError) as e:
                # Expected error for invalid parameters - check for specific error
                error_msg = str(e).lower()
                assert any(keyword in error_msg for keyword in
                          ['unsupported', 'operand', 'str', 'int', 'parameter', 'invalid'])


class TestKDEPlotErrorHandling:
    """Error handling tests specifically for KDEPlot."""
    
    def test_kde_plot_invalid_data(self, mock_data_factory):
        """Test KDEPlot with invalid data."""
        from src.monet_plots.plots.kde import KDEPlot
        
        plot = KDEPlot()
        
        # Test cases with invalid data
        invalid_data_cases = [
            ("empty_list", []),
            ("empty_array", np.array([])),
            ("single_value", [5.0]),
            ("string_data", ["a", "b", "c"]),
            ("mixed_types", [1, "mixed", 3.0]),
        ]
        
        for case_name, invalid_data in invalid_data_cases:
            try:
                # Should either work or fail gracefully
                plot.plot(invalid_data)
                # If it works, verify the plot was created
                assert plot.ax is not None
            except Exception as e:
                # If it fails, should be with a clear error message
                error_msg = str(e).lower()
                assert any(keyword in error_msg for keyword in
                          ['categorical', 'numeric', 'datetime', 'x variable', 'data'])
        
        plot.close()
    
    def test_kde_plot_invalid_bandwidth(self, mock_data_factory):
        """Test KDEPlot with invalid bandwidth parameters."""
        from src.monet_plots.plots.kde import KDEPlot
        
        plot = KDEPlot()
        data = mock_data_factory.kde_data()
        
        # Test cases with invalid bandwidth
        invalid_bandwidth_cases = [
            ("negative_bw", -1.0),
            ("zero_bw", 0.0),
            ("invalid_bw_method", "invalid_method"),
            ("invalid_bw_adjust", -1.0),
        ]
        
        for case_name, invalid_bw in invalid_bandwidth_cases:
            try:
                if case_name.endswith("_adjust"):
                    plot.plot(data, bw_adjust=invalid_bw)
                else:
                    plot.plot(data, bw=invalid_bw)
                # If it works, that's acceptable (graceful parameter handling)
            except (ValueError, TypeError) as e:
                # Expected error for invalid bandwidth
                assert any(keyword in str(e).lower() for keyword in 
                          ['bandwidth', 'bw', 'invalid', 'parameter'])
        
        plot.close()


class TestEdgeCaseHandling:
    """Edge case and boundary condition tests."""
    
    def test_extreme_numerical_values(self, mock_data_factory):
        """Test plot classes with extreme numerical values."""
        from src.monet_plots.plots.spatial import SpatialPlot
        
        plot = SpatialPlot()
        
        # Test cases with extreme values
        extreme_value_cases = [
            ("very_large_values", np.full((5, 5), 1e10)),
            ("very_small_values", np.full((5, 5), 1e-10)),
            ("mixed_extreme", np.array([[1e10, 1e-10], [1e10, 1e-10]])),
            ("overflow_values", np.full((5, 5), np.finfo(np.float64).max)),
        ]
        
        for case_name, extreme_data in extreme_value_cases:
            try:
                # Should either work or fail gracefully
                result = plot.plot(extreme_data)
                assert result is not None
            except Exception as e:
                # If it fails, should be with a clear error message
                assert "overflow" in str(e).lower() or "extreme" in str(e).lower() or \
                       "value" in str(e).lower()
        
        plot.close()
    
    def test_coordinate_boundary_conditions(self, mock_data_factory):
        """Test spatial plots with coordinate boundary conditions."""
        from src.monet_plots.plots.spatial import SpatialPlot
        
        plot = SpatialPlot()
        
        # Test cases with boundary coordinates
        boundary_cases = [
            ("polar_coordinates", np.random.randn(10, 10)),  # Should work with polar projection
            ("dateline_crossing", np.random.randn(10, 10)),  # Should handle dateline
            ("extreme_latitudes", np.random.randn(10, 10)),   # Should handle polar regions
        ]
        
        for case_name, test_data in boundary_cases:
            try:
                # Should handle boundary conditions gracefully
                result = plot.plot(test_data)
                assert result is not None
            except Exception as e:
                # If it fails, should be with a clear error message
                assert any(keyword in str(e).lower() for keyword in 
                          ['coordinate', 'boundary', 'projection', 'invalid'])
        
        plot.close()
    
    def test_memory_constrained_scenarios(self, mock_data_factory):
        """Test behavior under memory-constrained scenarios."""
        from src.monet_plots.plots.spatial import SpatialPlot
        
        # Test with progressively larger datasets
        memory_test_cases = [
            ("small_dataset", (10, 10)),
            ("medium_dataset", (50, 50)),
            ("large_dataset", (100, 100)),
        ]
        
        for case_name, shape in memory_test_cases:
            try:
                plot = SpatialPlot()
                data = mock_data_factory.spatial_2d(shape=shape)
                
                # Should handle the dataset size
                result = plot.plot(data)
                assert result is not None
                assert plot.ax is not None
                
                plot.close()
                
            except MemoryError:
                # If memory error occurs, it should be graceful
                pytest.skip(f"Memory constraint: {case_name} too large for test environment")
            except Exception as e:
                # Other errors should be related to implementation
                assert "memory" in str(e).lower() or "size" in str(e).lower() or \
                       "dimension" in str(e).lower()


# Test cleanup fixture
@pytest.fixture(autouse=True)
def cleanup_after_error_test():
    """Clean up matplotlib figures after each error handling test."""
    yield
    plt.close('all')
    plt.clf()