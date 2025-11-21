"""
Test configuration and shared fixtures for MONET Plots testing framework.

This module provides configuration, shared fixtures, and test utilities
that are used across all test modules.
"""

import pytest
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import Mock
import warnings
from pathlib import Path


# Test configuration
@pytest.fixture(scope="session")
def test_config():
    """Global test configuration."""
    return {
        'tolerance': 1e-10,
        'timeout': 30,
        'retry_attempts': 3,
        'random_seed': 42,
        'default_shape': (10, 10),
        'default_n_points': 100
    }


@pytest.fixture(scope="session")
def mock_data_factory(test_config):
    """Factory for creating mock data for different plot types."""
    class MockDataFactory:
        def __init__(self, config):
            self.config = config
            np.random.seed(config['random_seed'])
        
        def spatial_2d(self, shape=None, seed=None):
            """Generate 2D spatial data."""
            if seed is not None:
                np.random.seed(seed)
            else:
                np.random.seed(self.config['random_seed'])
            
            if shape is None:
                shape = self.config['default_shape']
            
            return np.random.randn(*shape)
        
        def time_series(self, n_points=None, start_date='2025-01-01', seed=None):
            """Generate time series data."""
            if seed is not None:
                np.random.seed(seed)
            else:
                np.random.seed(self.config['random_seed'])
            
            if n_points is None:
                n_points = self.config['default_n_points']
            
            dates = pd.date_range(start=start_date, periods=n_points, freq='D')
            # Create realistic time series with trend and noise
            trend = np.linspace(20, 25, n_points)
            seasonal = 5 * np.sin(2 * np.pi * np.arange(n_points) / 365 * 10)
            noise = np.random.randn(n_points) * 2
            values = trend + seasonal + noise
            
            model_values = values + np.random.randn(n_points) * 1.0
            
            return pd.DataFrame({
                'time': dates,
                'obs': values,
                'model': model_values,
                'units': 'ppb'
            })
        
        def scatter_data(self, n_points=None, correlation=0.8, seed=None):
            """Generate scatter plot data with specified correlation."""
            if seed is not None:
                np.random.seed(seed)
            else:
                np.random.seed(self.config['random_seed'])
            
            if n_points is None:
                n_points = self.config['default_n_points']
            
            x = np.random.randn(n_points)
            y = correlation * x + np.sqrt(1 - correlation**2) * np.random.randn(n_points)
            
            return pd.DataFrame({
                'x': x,
                'y': y,
                'category': np.random.choice(['A', 'B', 'C'], n_points)
            })
        
        def kde_data(self, n_points=None, distribution='normal', seed=None):
            """Generate data for KDE plot with different distributions."""
            if seed is not None:
                np.random.seed(seed)
            else:
                np.random.seed(self.config['random_seed'])
            
            if n_points is None:
                n_points = self.config['default_n_points']
            
            if distribution == 'normal':
                return np.random.randn(n_points)
            elif distribution == 'uniform':
                return np.random.uniform(-3, 3, n_points)
            elif distribution == 'bimodal':
                # Mix of two normal distributions
                data1 = np.random.randn(n_points // 2) - 2
                data2 = np.random.randn(n_points - n_points // 2) + 2
                return np.concatenate([data1, data2])
            else:
                return np.random.randn(n_points)
        
        def taylor_data(self, n_points=None, noise_level=0.3, seed=None):
            """Generate data for Taylor diagram."""
            if seed is not None:
                np.random.seed(seed)
            else:
                np.random.seed(self.config['random_seed'])
            
            if n_points is None:
                n_points = self.config['default_n_points']
            
            obs = np.random.randn(n_points) * 2 + 20
            model = obs + np.random.randn(n_points) * noise_level
            
            return pd.DataFrame({
                'obs': obs,
                'model': model
            })
        
        def spatial_dataframe(self, n_points=None, lat_range=(25, 50), lon_range=(-120, -70), seed=None):
            """Generate spatial point data."""
            if seed is not None:
                np.random.seed(seed)
            else:
                np.random.seed(self.config['random_seed'])
            
            if n_points is None:
                n_points = self.config['default_n_points'] // 2  # Smaller dataset for spatial
            
            return pd.DataFrame({
                'latitude': np.random.uniform(lat_range[0], lat_range[1], n_points),
                'longitude': np.random.uniform(lon_range[0], lon_range[1], n_points),
                'CMAQ': np.random.uniform(0, 50, n_points),
                'Obs': np.random.uniform(0, 50, n_points),
                'datetime': pd.to_datetime('2025-01-01')
            })
        
        def xarray_data(self, shape=None, lat_range=(25, 50), lon_range=(-120, -70), seed=None):
            """Generate xarray DataArray."""
            if seed is not None:
                np.random.seed(seed)
            else:
                np.random.seed(self.config['random_seed'])
            
            if shape is None:
                shape = self.config['default_shape']
            
            data = np.random.randn(*shape)
            lat = np.linspace(lat_range[0], lat_range[1], shape[0])
            lon = np.linspace(lon_range[0], lon_range[1], shape[1])
            
            return xr.DataArray(
                data,
                coords=[('latitude', lat), ('longitude', lon)],
                dims=['latitude', 'longitude']
            )
        
        def facet_data(self, seed=None):
            """Generate data for facet grid."""
            if seed is not None:
                np.random.seed(seed)
            else:
                np.random.seed(self.config['random_seed'])
            
            data = np.random.randn(3, 4, 5)
            return xr.DataArray(
                data,
                dims=['x', 'y', 'time'],
                coords={
                    'x': [1, 2, 3],
                    'y': [1, 2, 3, 4],
                    'time': [1, 2, 3, 4, 5]
                }
            )
        
        def wind_data(self, shape=None, seed=None):
            """Generate wind speed and direction data."""
            if seed is not None:
                np.random.seed(seed)
            else:
                np.random.seed(self.config['random_seed'])
            
            if shape is None:
                shape = self.config['default_shape']
            
            wind_speed = np.random.uniform(0, 20, shape)  # 0-20 m/s
            wind_dir = np.random.uniform(0, 360, shape)   # 0-360 degrees
            
            return wind_speed, wind_dir
    
    return MockDataFactory(test_config)


@pytest.fixture(scope="session")
def mock_grid_object():
    """Create a mock grid object for spatial plots."""
    gridobj = Mock()
    gridobj.variables = {
        'LAT': Mock(),
        'LON': Mock()
    }
    
    # Set up the mock to return reasonable coordinate arrays
    lat_data = np.linspace(25, 50, 10)
    lon_data = np.linspace(-120, -70, 10)
    
    gridobj.variables['LAT'][0, 0, :, :].squeeze.return_value = lat_data
    gridobj.variables['LON'][0, 0, :, :].squeeze.return_value = lon_data
    
    return gridobj


@pytest.fixture(scope="session")
def mock_basemap():
    """Create a mock basemap object."""
    m = Mock()
    
    # Mock the coordinate transformation
    def mock_transform(lon, lat):
        # Simple linear transformation for testing
        x = (lon + 180) * 100000  # Rough approximation
        y = lat * 100000
        return np.array([[x, x+1], [y, y+1]])
    
    m.side_effect = mock_transform
    m.drawstates = Mock()
    m.drawcoastlines = Mock()
    m.drawcountries = Mock()
    m.colorbar = Mock()
    m.imshow = Mock()
    m.contourf = Mock()
    m.quiver = Mock()
    m.barbs = Mock()
    
    return m


@pytest.fixture
def temp_directory():
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture(scope="function")
def cleanup_plots():
    """Clean up matplotlib figures after each test."""
    yield
    plt.close('all')
    plt.clf()


@pytest.fixture
def baseline_images_dir():
    """Directory for baseline images for visual regression tests."""
    return Path(__file__).parent / "baseline_images"


@pytest.fixture
def visual_thresholds():
    """Visual regression thresholds."""
    return {
        'pixel_tolerance': 0.01, # 1% of pixels can differ
        'structural_similarity': 0.95,  # SSIM threshold
        'color_tolerance': 10,  # RGB color difference tolerance
        'size_tolerance': 2 # Pixel size difference tolerance
    }


@pytest.fixture
def test_outputs_dir(temp_directory):
    """Directory for test output files."""
    outputs_dir = Path(temp_directory) / "test_outputs"
    outputs_dir.mkdir(exist_ok=True)
    return outputs_dir


# Performance test markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (may skip with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance benchmarks"
    )
    config.addinivalue_line(
        "markers", "visual: marks tests as visual regression tests"
    )


# Test data fixtures for specific scenarios
@pytest.fixture
def nan_data():
    """Create data with NaN values for testing NaN handling."""
    data = np.ones((5, 5))
    data[0, 0] = np.nan
    data[1, 1] = np.inf
    data[2, 2] = -np.inf
    return data


@pytest.fixture
def empty_dataframe():
    """Create empty DataFrame for testing error handling."""
    return pd.DataFrame()


@pytest.fixture
def single_point_data():
    """Create minimal data for testing edge cases."""
    return pd.DataFrame({
        'x': [1.0],
        'y': [2.0],
        'time': [pd.Timestamp('2025-01-01')]
    })


@pytest.fixture
def constant_data():
    """Create constant data for testing edge cases."""
    return np.ones((10, 10))


# Warning filters for cleaner test output
@pytest.fixture(autouse=True)
def filter_warnings():
    """Filter common warnings during testing."""
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    yield
    warnings.resetwarnings()


# Error simulation fixtures
@pytest.fixture
def permission_error_simulation():
    """Simulate permission errors for testing error handling."""
    def mock_permission_error(*args, **kwargs):
        raise PermissionError("Simulated permission error")
    return mock_permission_error


@pytest.fixture
def file_not_found_simulation():
    """Simulate file not found errors for testing error handling."""
    def mock_file_not_found(*args, **kwargs):
        raise FileNotFoundError("Simulated file not found")
    return mock_file_not_found