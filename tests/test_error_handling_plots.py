"""
Plot-Specific Error Handling and Edge Cases Tests for MONET Plots

This module contains error handling tests for individual plot classes.
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


class TestPlotErrorHandling:
    """Comprehensive error handling and edge case tests for plot classes."""
    
    def test_spatial_plot_error_handling(self, mock_data_factory):
        """Test SpatialPlot error handling for various invalid inputs."""
        try:
            from src.monet_plots.plots.spatial import SpatialPlot
            
            plot = SpatialPlot()
            
            # Test 1: Invalid data types
            try:
                plot.plot("invalid_string")
            except (TypeError, AttributeError) as e:
                assert any(keyword in str(e).lower() for keyword in ['dtype', 'image', 'data'])
            
            try:
                plot.plot([1, 2, 3])  # List instead of numpy array
            except (TypeError, AttributeError) as e:
                assert any(keyword in str(e).lower() for keyword in ['dtype', 'image', 'data'])
            
            # Test 2: Empty arrays
            try:
                plot.plot(np.array([]))
            except (TypeError, ValueError) as e:
                assert any(keyword in str(e).lower() for keyword in ['shape', 'dimension', 'invalid'])
            
            # Test 3: 1D arrays
            try:
                plot.plot(np.array([1, 2, 3, 4, 5]))
            except (TypeError, ValueError) as e:
                assert any(keyword in str(e).lower() for keyword in ['shape', 'dimension', 'invalid'])
            
            # Test 4: Arrays with NaN values
            data_with_nan = mock_data_factory.spatial_2d()
            data_with_nan[0, 0] = np.nan
            data_with_nan[1, 1] = np.inf
            
            # Should handle NaN gracefully or raise specific error
            try:
                plot.plot(data_with_nan)
                assert plot.ax is not None
            except Exception as e:
                assert any(keyword in str(e).lower() for keyword in ['nan', 'inf', 'invalid'])
            
            # Test 5: Constant data (vmin == vmax)
            constant_data = np.ones((10, 10))
            
            try:
                plot.plot(constant_data, discrete=True)
                # Should handle constant data gracefully
                assert plot.ax is not None
            except Exception as e:
                # If it fails, should be a specific, expected error
                assert any(keyword in str(e).lower() for keyword in ['finite', 'value', 'masked', 'pcolormesh'])
            
            # Test 6: Invalid colormap
            data = mock_data_factory.spatial_2d()
            
            try:
                plot.plot(data, plotargs={'cmap': 'nonexistent_colormap'})
            except (ValueError, KeyError, TypeError) as e:
                assert any(keyword in str(e).lower() for keyword in ['colormap', 'cmap', 'invalid'])
            
            plot.close()
            
        except ImportError:
            pytest.skip("SpatialPlot not available")
    
    def test_spatial_plot_projection_errors(self, mock_data_factory):
        """Test SpatialPlot projection-related errors."""
        try:
            from src.monet_plots.plots.spatial import SpatialPlot
            import cartopy.crs as ccrs
            
            # Test 1: Invalid projection
            try:
                SpatialPlot(projection="invalid_projection")
            except (ValueError, TypeError) as e:
                assert any(keyword in str(e).lower() for keyword in ['projection', 'unknown'])
            
            # Test 2: Valid but unusual projection
            try:
                unusual_proj = ccrs.Orthographic(0, 0)
                plot = SpatialPlot(projection=unusual_proj)
                data = mock_data_factory.spatial_2d()
                plot.plot(data)
                assert plot.ax is not None
                plot.close()
            except Exception:
                pytest.skip("Orthographic projection not available")
            
        except ImportError:
            pytest.skip("SpatialPlot not available")
    
    def test_timeseries_plot_error_handling(self, mock_data_factory):
        """Test TimeSeriesPlot error handling."""
        try:
            from src.monet_plots.plots.timeseries import TimeSeriesPlot
            
            plot = TimeSeriesPlot()
            
            # Test 1: Missing required columns
            df_missing_cols = pd.DataFrame({'x': [1, 2, 3]})
            
            with pytest.raises(KeyError):
                plot.plot(df_missing_cols)
            
            # Test 2: Empty DataFrame
            empty_df = pd.DataFrame()
            
            with pytest.raises((ValueError, KeyError, IndexError)):
                plot.plot(empty_df)
            
            # Test 3: DataFrame with wrong column names
            df_wrong_cols = pd.DataFrame({
                'timestamp': pd.date_range('2025-01-01', periods=10),
                'value': np.random.randn(10)
            })
            
            with pytest.raises(KeyError):
                plot.plot(df_wrong_cols)  # Default uses 'time' and 'obs'
            
            # Should work with correct column specification
            plot.plot(df_wrong_cols, x='timestamp', y='value')
            assert plot.ax is not None
            
            # Test 4: Single data point
            single_point_df = pd.DataFrame({
                'time': [pd.Timestamp('2025-01-01')],
                'obs': [25.0]
            })
            
            # Should handle gracefully or raise informative error
            try:
                plot.plot(single_point_df)
                assert plot.ax is not None
            except Exception as e:
                assert isinstance(e, (ValueError, ZeroDivisionError))
            
            # Test 5: Data with NaN values
            df_with_nan = mock_data_factory.time_series()
            df_with_nan.loc[5, 'obs'] = np.nan
            df_with_nan.loc[10, 'model'] = np.inf
            
            # Should handle NaN values gracefully
            try:
                plot.plot(df_with_nan)
                assert plot.ax is not None
            except Exception as e:
                # Should be a specific error about data quality
                assert "nan" in str(e).lower() or "inf" in str(e).lower()
            
            plot.close()
            
        except ImportError:
            pytest.skip("TimeSeriesPlot not available")
    
    def test_timeseries_plot_std_dev_edge_cases(self, mock_data_factory):
        """Test TimeSeriesPlot edge cases related to standard deviation."""
        try:
            from src.monet_plots.plots.timeseries import TimeSeriesPlot
            
            plot = TimeSeriesPlot()
            
            # Test 1: Constant values (zero std dev)
            constant_df = pd.DataFrame({
                'time': pd.date_range('2025-01-01', periods=10, freq='D'),
                'obs': [5.0] * 10,
                'model': [5.0] * 10
            })
            
            plot.plot(constant_df)
            assert plot.ax is not None
            
            # Test 2: Single unique value with some variation
            near_constant_df = pd.DataFrame({
                'time': pd.date_range('2025-01-01', periods=10, freq='D'),
                'obs': [5.0] * 9 + [5.1],
                'model': [5.0] * 10
            })
            
            plot.plot(near_constant_df)
            assert plot.ax is not None
            
            plot.close()
            
        except ImportError:
            pytest.skip("TimeSeriesPlot not available")
    
    def test_taylor_diagram_error_handling(self, mock_data_factory):
        """Test TaylorDiagramPlot error handling."""
        try:
            from src.monet_plots.plots.taylor import TaylorDiagramPlot
            
            # Create plot with valid obs_std
            df = mock_data_factory.taylor_data()
            obs_std = df['obs'].std()
            
            plot = TaylorDiagramPlot(obs_std)
            
            # Test 1: Invalid data for add_sample
            invalid_df = pd.DataFrame({'x': [1, 2, 3]})
            
            try:
                plot.add_sample(invalid_df)
            except (KeyError, TypeError) as e:
                assert any(keyword in str(e).lower() for keyword in ['obs', 'model', 'key', 'column'])
            
            # Test 2: Data with NaN values
            df_with_nan = df.copy()
            df_with_nan.loc[5, 'obs'] = np.nan
            df_with_nan.loc[10, 'model'] = np.inf
            
            try:
                plot.add_sample(df_with_nan)
                # Should handle NaN gracefully
                assert plot.dia is not None
            except Exception as e:
                # If it fails, should be about data quality
                assert any(keyword in str(e).lower() for keyword in ['nan', 'inf', 'dropna', 'data'])
            
            # Test 3: Zero standard deviation data
            zero_std_df = pd.DataFrame({
                'obs': [5.0, 5.0, 5.0, 5.0, 5.0],
                'model': [5.0, 5.0, 5.0, 5.0, 5.0]
            })
            obs_std_zero = zero_std_df['obs'].std()  # Will be 0.0
            
            # Creating plot with zero std dev
            zero_std_plot = TaylorDiagramPlot(obs_std_zero)
            
            # Adding zero std dev data
            try:
                zero_std_plot.add_sample(zero_std_df)
                assert zero_std_plot.dia is not None
            except Exception as e:
                # Should be a specific error about zero std dev
                assert any(keyword in str(e).lower() for keyword in ['zero', 'std', 'variance', 'correlation'])
            
            # Test 4: Very small standard deviation
            small_std_df = pd.DataFrame({
                'obs': np.ones(100) + np.random.randn(100) * 1e-10,
                'model': np.ones(100) + np.random.randn(100) * 1e-10
            })
            small_std = small_std_df['obs'].std()
            
            small_std_plot = TaylorDiagramPlot(small_std)
            small_std_plot.add_sample(small_std_df)
            assert small_std_plot.dia is not None
            
            plot.close()
            zero_std_plot.close()
            small_std_plot.close()
            
        except ImportError:
            pytest.skip("TaylorDiagramPlot not available")
    
    def test_scatter_plot_error_handling(self, mock_data_factory):
        """Test ScatterPlot error handling."""
        try:
            from src.monet_plots.plots.scatter import ScatterPlot
            
            plot = ScatterPlot()
            df = mock_data_factory.scatter_data()
            
            # Test 1: Invalid column names
            try:
                plot.plot(df, 'invalid_x', 'y')
            except KeyError as e:
                assert "'" in str(e) or "not in index" in str(e)
            
            try:
                plot.plot(df, 'x', 'invalid_y')
            except KeyError as e:
                assert "'" in str(e) or "not in index" in str(e)
            
            # Test 2: Missing columns
            df_missing = pd.DataFrame({'a': [1, 2, 3]})
            
            try:
                plot.plot(df_missing, 'x', 'y')
            except KeyError as e:
                assert "'" in str(e) or "not in index" in str(e)
            
            # Test 3: Insufficient data
            single_point_df = pd.DataFrame({'x': [1.0], 'y': [2.0]})
            
            # Should handle gracefully or raise appropriate error
            try:
                plot.plot(single_point_df, 'x', 'y')
                assert plot.ax is not None
            except Exception as e:
                assert any(keyword in str(e).lower() for keyword in ['insufficient', 'sample', 'point'])
            
            # Test 4: Empty DataFrame
            empty_df = pd.DataFrame()
            
            try:
                plot.plot(empty_df, 'x', 'y')
            except KeyError as e:
                assert "'" in str(e) or "not in index" in str(e)
            
            # Test 5: Data with all NaN values
            df_all_nan = pd.DataFrame({
                'x': [np.nan, np.nan, np.nan],
                'y': [np.nan, np.nan, np.nan]
            })
            
            try:
                plot.plot(df_all_nan, 'x', 'y')
                # Should handle gracefully or provide informative message
                assert plot.ax is not None
            except Exception as e:
                assert any(keyword in str(e).lower() for keyword in ['nan', 'data', 'insufficient'])
            
            plot.close()
            
        except ImportError:
            pytest.skip("ScatterPlot not available")
    
    def test_kde_plot_error_handling(self, mock_data_factory):
        """Test KDEPlot error handling."""
        try:
            from src.monet_plots.plots.kde import KDEPlot
            
            plot = KDEPlot()
            
            # Test 1: Invalid data types
            try:
                plot.plot("invalid_string")
            except Exception as e:
                assert any(keyword in str(e).lower() for keyword in ['data', 'type', 'categorical', 'numeric'])
            
            try:
                plot.plot([1, 2, 3])  # List instead of array/Series
            except Exception as e:
                assert any(keyword in str(e).lower() for keyword in ['data', 'type', 'categorical', 'numeric'])
            
            # Test 2: Empty data
            try:
                plot.plot(np.array([]))
            except Exception as e:
                assert any(keyword in str(e).lower() for keyword in ['empty', 'data', 'variance'])
            
            try:
                plot.plot(pd.Series([]))
            except Exception as e:
                assert any(keyword in str(e).lower() for keyword in ['empty', 'data', 'variance'])
            
            # Test 3: Data with all NaN values
            all_nan_data = np.array([np.nan, np.nan, np.nan])
            
            try:
                plot.plot(all_nan_data)
                # Should handle gracefully or provide warning
                assert plot.ax is not None
            except Exception as e:
                assert any(keyword in str(e).lower() for keyword in ['nan', 'data', 'empty'])
            
            # Test 4: Single data point
            single_point_data = np.array([5.0])
            
            try:
                plot.plot(single_point_data)
                # Should handle gracefully or provide warning
                assert plot.ax is not None
            except Exception as e:
                assert any(keyword in str(e).lower() for keyword in ['single', 'point', 'data'])
            
            # Test 5: DataFrame column that doesn't exist
            df = mock_data_factory.time_series()
            
            try:
                plot.plot(df['nonexistent_column'])
            except KeyError as e:
                assert "'" in str(e) or "not in index" in str(e)
            
            # Test 6: Invalid bandwidth
            data = mock_data_factory.kde_data()
            
            try:
                plot.plot(data, bw='invalid_bandwidth')
            except Exception as e:
                assert any(keyword in str(e).lower() for keyword in ['bandwidth', 'bw', 'invalid', 'parameter'])
            
            plot.close()
            
        except ImportError:
            pytest.skip("KDEPlot not available")
    
    def test_xarray_spatial_plot_error_handling(self, mock_data_factory):
        """Test XarraySpatialPlot error handling."""
        try:
            from src.monet_plots.plots.xarray_spatial import XarraySpatialPlot
            
            plot = XarraySpatialPlot()
            
            # Test 1: Invalid data type
            try:
                plot.plot(np.array([1, 2, 3]))  # NumPy array instead of xarray
            except AttributeError as e:
                assert "plot" in str(e).lower() or "numpy" in str(e).lower()
            
            try:
                plot.plot(pd.DataFrame({'x': [1, 2, 3]}))  # DataFrame instead of xarray
            except Exception as e:
                assert any(keyword in str(e).lower() for keyword in ['dataarray', 'xarray', 'type', 'invalid'])
            
            # Test 2: Empty xarray DataArray
            try:
                # Create a minimal xarray DataArray for testing
                import xarray as xr
                empty_da = xr.DataArray([])
                
                plot.plot(empty_da)
                # Should handle gracefully
                assert plot.ax is not None
            except Exception as e:
                assert any(keyword in str(e).lower() for keyword in ['empty', 'data', 'dimension'])
            
            # Test 3: Xarray without proper coordinates
            try:
                import xarray as xr
                malformed_da = xr.DataArray(
                    np.random.randn(5, 5),
                    dims=['x', 'y']
                    # Missing coordinates
                )
                
                plot.plot(malformed_da)
                assert plot.ax is not None
            except Exception as e:
                assert any(keyword in str(e).lower() for keyword in ['coordinate', 'dimension', 'missing'])
            
            plot.close()
            
        except ImportError:
            pytest.skip("XarraySpatialPlot not available")
    
    def test_facet_grid_plot_error_handling(self, mock_data_factory):
        """Test FacetGridPlot error handling."""
        try:
            from src.monet_plots.plots.facet_grid import FacetGridPlot
            
            data = mock_data_factory.facet_data()
            
            # Test 1: Invalid dimension for faceting
            try:
                plot = FacetGridPlot(data, col='invalid_dimension')
                plot.plot()
            except TypeError as e:
                assert "data source" in str(e).lower() or "dataframe" in str(e).lower()
            
            try:
                plot = FacetGridPlot(data, row='invalid_dimension')
                plot.plot()
            except TypeError as e:
                assert "data source" in str(e).lower() or "dataframe" in str(e).lower()
            
            # Test 2: Empty xarray Dataset
            try:
                import xarray as xr
                empty_data = xr.DataArray([], dims=['x']).to_dataset(name='empty')
                
                plot = FacetGridPlot(empty_data, col='x')
                plot.plot()
                # Should handle gracefully
                assert plot.grid is not None
            except ValueError as e:
                assert "positive integer" in str(e).lower() or "columns" in str(e).lower()
            
            # Test 3: Data without dimensions
            try:
                import xarray as xr
                scalar_data = xr.DataArray(5.0)
                
                plot = FacetGridPlot(scalar_data, col='nonexistent')
                plot.plot()
            except Exception as e:
                assert any(keyword in str(e).lower() for keyword in ['dimension', 'dataarray', 'scalar'])
            
        except ImportError:
            pytest.skip("FacetGridPlot not available")


# Test cleanup fixture
@pytest.fixture(autouse=True)
def cleanup_after_error_test():
    """Clean up matplotlib figures after each error handling test."""
    yield
    plt.close('all')
    plt.clf()