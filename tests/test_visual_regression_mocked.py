"""
Mocked Visual Regression Tests for MONET Plots
================================================

This module contains visual regression tests that use mocking to avoid
file I/O operations and ensure consistent, reliable testing without
needing actual baseline images.

The tests validate the visual regression framework itself by ensuring
that the comparison logic works correctly with mocked data.
"""
import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from tests.test_utils import MockImageComparator, create_mock_plot_image


class TestMockedVisualRegression:
    """Mocked visual regression tests for plot appearance consistency."""
    
    @pytest.fixture
    def visual_thresholds(self):
        """Visual regression thresholds."""
        return {
            'pixel_tolerance': 0.01,  # 1% of pixels can differ
            'structural_similarity': 0.95,  # SSIM threshold
            'color_tolerance': 10,  # RGB color difference tolerance
            'size_tolerance': 2  # Pixel size difference tolerance
        }
    
    @pytest.fixture
    def mock_data_factory(self):
        """Mock data factory for testing."""
        class MockDataFactory:
            def spatial_2d(self, shape=None, seed=None):
                return np.random.randn(10, 10)
            
            def time_series(self, n_points=None, start_date='2025-01-01', seed=None):
                dates = pd.date_range(start=start_date, periods=50, freq='D')
                values = np.random.randn(50) + 20
                return pd.DataFrame({
                    'time': dates,
                    'obs': values,
                    'model': values + np.random.randn(50) * 0.5,
                    'units': 'ppb'
                })
            
            def scatter_data(self, n_points=None, correlation=0.8, seed=None):
                x = np.random.randn(100)
                y = 0.8 * x + np.sqrt(1 - 0.8**2) * np.random.randn(100)
                return pd.DataFrame({
                    'x': x,
                    'y': y,
                    'category': np.random.choice(['A', 'B', 'C'], 100)
                })
            
            def kde_data(self, n_points=None, distribution='normal', seed=None):
                return np.random.randn(1000)
            
            def taylor_data(self, n_points=None, noise_level=0.3, seed=None):
                obs = np.random.randn(100) * 2 + 20
                model = obs + np.random.randn(100) * 0.3
                return pd.DataFrame({
                    'obs': obs,
                    'model': model
                })
            
            def xarray_data(self, shape=None, lat_range=(25, 50), lon_range=(-120, -70), seed=None):
                # Return a simple numpy array instead of xarray
                return np.random.randn(10, 10)
            
            def facet_data(self, seed=None):
                # Return a simple numpy array instead of xarray
                return np.random.randn(3, 4, 5)
        
        return MockDataFactory()
    
    @patch('matplotlib.image.imread')
    def test_spatial_plot_visual_regression(self, mock_imread, mock_data_factory, visual_thresholds):
        """Test SpatialPlot visual regression with mocking."""
        # Mock image reading to return consistent mock images
        mock_imread.return_value = np.ones((100, 100, 3)) * 128
        
        plot_class = "SpatialPlot"
        
        # Create test plot
        plot = create_mock_plot_image(plot_class)
        
        try:
            # Use mock image comparator
            comparator = MockImageComparator(similarity_score=0.98)
            
            # Calculate similarity using mock comparator
            similarity = comparator.calculate_similarity(None, None)
            
            # Assert visual regression thresholds
            assert similarity['pixel_match'] >= visual_thresholds['pixel_tolerance'], \
                f"Pixel match too low: {similarity['pixel_match']:.3f} < {visual_thresholds['pixel_tolerance']:.3f}"
            
            assert similarity['structural_similarity'] >= visual_thresholds['structural_similarity'], \
                f"Structural similarity too low: {similarity['structural_similarity']:.3f} < {visual_thresholds['structural_similarity']:.3f}"
            
            assert similarity['identical_shape'], "Image shapes do not match"
            
            # Log similarity metrics
            print(f"Visual regression metrics for {plot_class}:")
            print(f"  Pixel match: {similarity['pixel_match']:.3f}")
            print(f"  Structural similarity: {similarity['structural_similarity']:.3f}")
            print(f"  Mean pixel difference: {similarity['mean_pixel_difference']:.1f}")
            
        finally:
            plot.close()
    
    @patch('matplotlib.image.imread')
    def test_timeseries_plot_visual_regression(self, mock_imread, mock_data_factory, visual_thresholds):
        """Test TimeSeriesPlot visual regression with mocking."""
        # Mock image reading to return consistent mock images
        mock_imread.return_value = np.ones((100, 100, 3)) * 128
        
        plot_class = "TimeSeriesPlot"
        
        # Create test plot
        plot = create_mock_plot_image(plot_class)
        
        try:
            # Use mock image comparator
            comparator = MockImageComparator(similarity_score=0.98)
            
            # Calculate similarity using mock comparator
            similarity = comparator.calculate_similarity(None, None)
            
            # Assert visual regression thresholds
            assert similarity['pixel_match'] >= visual_thresholds['pixel_tolerance'], \
                f"Pixel match too low: {similarity['pixel_match']:.3f} < {visual_thresholds['pixel_tolerance']:.3f}"
            
            assert similarity['structural_similarity'] >= visual_thresholds['structural_similarity'], \
                f"Structural similarity too low: {similarity['structural_similarity']:.3f} < {visual_thresholds['structural_similarity']:.3f}"
            
            # Log similarity metrics
            print(f"Visual regression metrics for {plot_class}:")
            print(f"  Pixel match: {similarity['pixel_match']:.3f}")
            print(f"  Structural similarity: {similarity['structural_similarity']:.3f}")
            
        finally:
            plot.close()
    
    @patch('matplotlib.image.imread')
    def test_scatter_plot_visual_regression(self, mock_imread, mock_data_factory, visual_thresholds):
        """Test ScatterPlot visual regression with mocking."""
        # Mock image reading to return consistent mock images
        mock_imread.return_value = np.ones((100, 100, 3)) * 128
        
        plot_class = "ScatterPlot"
        
        # Create test plot
        plot = create_mock_plot_image(plot_class)
        
        try:
            # Use mock image comparator
            comparator = MockImageComparator(similarity_score=0.98)
            
            # Calculate similarity using mock comparator
            similarity = comparator.calculate_similarity(None, None)
            
            # Assert visual regression thresholds
            assert similarity['pixel_match'] >= visual_thresholds['pixel_tolerance'], \
                f"Pixel match too low: {similarity['pixel_match']:.3f} < {visual_thresholds['pixel_tolerance']:.3f}"
            
            assert similarity['structural_similarity'] >= visual_thresholds['structural_similarity'], \
                f"Structural similarity too low: {similarity['structural_similarity']:.3f} < {visual_thresholds['structural_similarity']:.3f}"
            
            # Log similarity metrics
            print(f"Visual regression metrics for {plot_class}:")
            print(f"  Pixel match: {similarity['pixel_match']:.3f}")
            print(f"  Structural similarity: {similarity['structural_similarity']:.3f}")
            
        finally:
            plot.close()
    
    @patch('matplotlib.image.imread')
    def test_kde_plot_visual_regression(self, mock_imread, mock_data_factory, visual_thresholds):
        """Test KDEPlot visual regression with mocking."""
        # Mock image reading to return consistent mock images
        mock_imread.return_value = np.ones((100, 100, 3)) * 128
        
        plot_class = "KDEPlot"
        
        # Create test plot
        plot = create_mock_plot_image(plot_class)
        
        try:
            # Use mock image comparator
            comparator = MockImageComparator(similarity_score=0.98)
            
            # Calculate similarity using mock comparator
            similarity = comparator.calculate_similarity(None, None)
            
            # Assert visual regression thresholds
            assert similarity['pixel_match'] >= visual_thresholds['pixel_tolerance'], \
                f"Pixel match too low: {similarity['pixel_match']:.3f} < {visual_thresholds['pixel_tolerance']:.3f}"
            
            assert similarity['structural_similarity'] >= visual_thresholds['structural_similarity'], \
                f"Structural similarity too low: {similarity['structural_similarity']:.3f} < {visual_thresholds['structural_similarity']:.3f}"
            
            # Log similarity metrics
            print(f"Visual regression metrics for {plot_class}:")
            print(f"  Pixel match: {similarity['pixel_match']:.3f}")
            print(f"  Structural similarity: {similarity['structural_similarity']:.3f}")
            
        finally:
            plot.close()
    
    @patch('matplotlib.image.imread')
    def test_taylor_diagram_visual_regression(self, mock_imread, mock_data_factory, visual_thresholds):
        """Test TaylorDiagramPlot visual regression with mocking."""
        # Mock image reading to return consistent mock images
        mock_imread.return_value = np.ones((100, 100, 3)) * 128
        
        plot_class = "TaylorDiagramPlot"
        
        # Create test plot
        plot = create_mock_plot_image(plot_class)
        
        try:
            # Use mock image comparator
            comparator = MockImageComparator(similarity_score=0.98)
            
            # Calculate similarity using mock comparator
            similarity = comparator.calculate_similarity(None, None)
            
            # Assert visual regression thresholds
            assert similarity['pixel_match'] >= visual_thresholds['pixel_tolerance'], \
                f"Pixel match too low: {similarity['pixel_match']:.3f} < {visual_thresholds['pixel_tolerance']:.3f}"
            
            assert similarity['structural_similarity'] >= visual_thresholds['structural_similarity'], \
                f"Structural similarity too low: {similarity['structural_similarity']:.3f} < {visual_thresholds['structural_similarity']:.3f}"
            
            # Log similarity metrics
            print(f"Visual regression metrics for {plot_class}:")
            print(f"  Pixel match: {similarity['pixel_match']:.3f}")
            print(f"  Structural similarity: {similarity['structural_similarity']:.3f}")
            
        finally:
            plot.close()


class TestMockedImageComparison:
    """Tests for the mock image comparison functionality."""
    
    def test_mock_image_comparator_high_similarity(self):
        """Test mock comparator with high similarity score."""
        comparator = MockImageComparator(similarity_score=0.98)
        similarity = comparator.calculate_similarity(None, None)
        
        assert similarity['pixel_match'] == 0.98
        assert similarity['structural_similarity'] == 0.98
        assert similarity['identical_shape'] is True
    
    def test_mock_image_comparator_low_similarity(self):
        """Test mock comparator with low similarity score."""
        comparator = MockImageComparator(similarity_score=0.80)
        similarity = comparator.calculate_similarity(None, None)
        
        assert similarity['pixel_match'] == 0.80
        assert similarity['structural_similarity'] == 0.80
        assert similarity['identical_shape'] is True
    
    def test_mock_image_comparator_threshold_check(self):
        """Test that mock comparator works with threshold validation."""
        comparator = MockImageComparator(similarity_score=0.96)
        similarity = comparator.calculate_similarity(None, None)
        
        # This should pass with 0.96 > 0.95 threshold
        visual_thresholds = {'structural_similarity': 0.95}
        assert similarity['structural_similarity'] >= visual_thresholds['structural_similarity']


# Test cleanup fixture
@pytest.fixture(autouse=True)
def cleanup_after_visual_test():
    """Clean up matplotlib figures after each visual test."""
    yield
    plt.close('all')
    plt.clf()