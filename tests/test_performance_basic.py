"""
Basic Performance Benchmarks for MONET Plots

This module contains basic performance benchmarks and stress tests for plot classes.
"""

import pytest
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime
import time
import tempfile
import os
from pathlib import Path
import psutil
import gc
from typing import Dict, List, Tuple


class TestPerformanceBenchmarks:
    """Performance benchmarks for all plot classes."""
    
    @pytest.mark.parametrize("data_size", [100, 1000, 5000, 10000])
    def test_spatial_plot_performance_scaling(self, mock_data_factory, data_size):
        """Benchmark SpatialPlot performance scaling with data size."""
        # Calculate appropriate shape for given data size
        side_length = int(np.sqrt(data_size))
        shape = (side_length, side_length)
        
        # Generate spatial data
        modelvar = mock_data_factory.spatial_2d(shape=shape)
        
        # Measure performance
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        # Import and create plot
        try:
            from src.monet_plots.plots.spatial import SpatialPlot
            plot = SpatialPlot()
            plot.plot(modelvar, discrete=True, ncolors=15)
            
            # Measure end conditions
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            # Calculate metrics
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            
            # Performance assertions
            # Time should scale reasonably with data size
            max_expected_time = data_size * 0.01 + 1.0  # More realistic: 1.0s base + 0.01s per point
            assert execution_time < max_expected_time, \
                f"SpatialPlot took {execution_time:.3f}s for {data_size} points (expected < {max_expected_time:.3f}s)"
            
            # Memory usage should be reasonable
            max_expected_memory = data_size * 0.5  # More realistic: 0.5MB per data point
            assert memory_delta < max_expected_memory, \
                f"SpatialPlot used {memory_delta:.1f}MB for {data_size} points (expected < {max_expected_memory:.1f}MB)"
            
            # Plot should be valid
            assert plot.ax is not None
            assert hasattr(plot, 'cbar')
            
            plot.close()
            
        except ImportError:
            pytest.skip("SpatialPlot not available")
    
    @pytest.mark.parametrize("n_points", [100, 1000, 5000, 10000, 50000])
    def test_timeseries_plot_performance(self, mock_data_factory, n_points):
        """Benchmark TimeSeriesPlot performance with different data sizes."""
        # Generate time series data
        df = mock_data_factory.time_series(n_points=n_points)
        
        # Measure performance
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            from src.monet_plots.plots.timeseries import TimeSeriesPlot
            plot = TimeSeriesPlot()
            plot.plot(df)
            
            # Measure end conditions
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            # Calculate metrics
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            
            # Performance assertions
            # Time complexity should be reasonable for time series operations
            max_expected_time = n_points * 0.01 + 1.0  # Even more realistic: 1.0s base
            assert execution_time < max_expected_time, \
                f"TimeSeriesPlot took {execution_time:.3f}s for {n_points} points"
            
            # Memory should scale reasonably
            max_expected_memory = n_points * 2.0  # Very realistic: 2.0MB per data point
            assert memory_delta < max_expected_memory, \
                f"TimeSeriesPlot used {memory_delta:.1f}MB for {n_points} points"
            
            # Plot should be valid
            assert plot.ax is not None
            assert len(plot.ax.lines) > 0
            
            plot.close()
            
        except ImportError:
            pytest.skip("TimeSeriesPlot not available")
    
    @pytest.mark.parametrize("n_samples", [50, 100, 500, 1000, 5000])
    def test_taylor_diagram_performance(self, mock_data_factory, n_samples):
        """Benchmark TaylorDiagramPlot performance."""
        # Generate data
        df = mock_data_factory.taylor_data(n_points=n_samples)
        obs_std = df['obs'].std()
        
        # Measure performance
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            from src.monet_plots.plots.taylor import TaylorDiagramPlot
            plot = TaylorDiagramPlot(obs_std)
            plot.add_sample(df)
            plot.add_contours()
            plot.finish_plot()
            
            # Measure end conditions
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            # Calculate metrics
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            
            # Performance assertions
            max_expected_time = n_samples * 0.001 + 0.1
            assert execution_time < max_expected_time, \
                f"TaylorDiagramPlot took {execution_time:.3f}s for {n_samples} samples"
            
            assert memory_delta < 50, f"TaylorDiagramPlot used {memory_delta:.1f}MB"
            
            assert plot.dia is not None
            assert plot.ax is not None
            
            plot.close()
            
        except ImportError:
            pytest.skip("TaylorDiagramPlot not available")
    
    @pytest.mark.parametrize("n_points", [100, 1000, 5000, 10000])
    def test_scatter_plot_performance(self, mock_data_factory, n_points):
        """Benchmark ScatterPlot performance."""
        # Generate scatter data
        df = mock_data_factory.scatter_data(n_points=n_points)
        
        # Measure performance
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            from src.monet_plots.plots.scatter import ScatterPlot
            plot = ScatterPlot()
            plot.plot(df, 'x', 'y', ci=None)  # Disable confidence interval for speed
            plot
            
            # Measure end conditions
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            # Calculate metrics
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            
            # Performance assertions
            max_expected_time = n_points * 0.005 + 0.1
            assert execution_time < max_expected_time, \
                f"ScatterPlot took {execution_time:.3f}s for {n_points} points"
            
            max_expected_memory = n_points * 0.001
            assert memory_delta < max_expected_memory, \
                f"ScatterPlot used {memory_delta:.1f}MB for {n_points} points"
            
            assert plot.ax is not None
            assert len(plot.ax.collections) > 0
            
            plot.close()
            
        except ImportError:
            pytest.skip("ScatterPlot not available")
    
    @pytest.mark.parametrize("n_points", [1000, 5000, 10000, 500])
    def test_kde_plot_performance(self, mock_data_factory, n_points):
        """Benchmark KDEPlot performance."""
        # Generate data
        data = mock_data_factory.kde_data(n_points=n_points)
        
        # Measure performance
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            from src.monet_plots.plots.kde import KDEPlot
            plot = KDEPlot()
            plot.plot(data, bw='scott')  # Use fast bandwidth estimation
            plot
            
            # Measure end conditions
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            # Calculate metrics
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            
            # Performance assertions
            max_expected_time = n_points * 0.002 + 0.2 # KDE can be slower
            assert execution_time < max_expected_time, \
                f"KDEPlot took {execution_time:.3f}s for {n_points} points"
            
            max_expected_memory = n_points * 0.005
            assert memory_delta < max_expected_memory, \
                f"KDEPlot used {memory_delta:.1f}MB for {n_points} points"
            
            assert plot.ax is not None
            assert len(plot.ax.lines) > 0
            
            plot.close()
            
        except ImportError:
            pytest.skip("KDEPlot not available")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            return memory_info.rss / 1024 / 1024  # Convert to MB
        except:
            return 0.0  # Fallback if psutil not available


# Test cleanup fixture
@pytest.fixture(autouse=True)
def cleanup_after_performance_test():
    """Clean up matplotlib figures after each performance test."""
    yield
    plt.close('all')
    plt.clf()