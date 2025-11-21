"""
Test utilities for MONET Plots testing framework.

This module provides utilities for mocking, image comparison, and test infrastructure.
"""
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import matplotlib.pyplot as plt


class MockImageComparator:
    """
    Mock image comparator that simulates image comparison without requiring actual files.
    """
    
    def __init__(self, similarity_score=0.98):
        """
        Initialize the mock comparator with a default similarity score.
        
        Args:
            similarity_score (float): Default similarity score to return (0-1 range)
        """
        self.similarity_score = similarity_score
    
    def calculate_similarity(self, img1, img2):
        """
        Calculate mock similarity metrics between two images.
        
        Args:
            img1: First image (ignored in mock)
            img2: Second image (ignored in mock)
        
        Returns:
            dict: Mock similarity metrics
        """
        return {
            'pixel_match': self.similarity_score,
            'structural_similarity': self.similarity_score,
            'mean_pixel_difference': 1.0 - self.similarity_score,
            'max_pixel_difference': 2.0 * (1.0 - self.similarity_score),
            'identical_shape': True
        }
    
    def imread(self, path):
        """
        Mock image reading that returns a consistent mock array.
        
        Args:
            path: Path to image (ignored in mock)
        
        Returns:
            np.ndarray: Mock image array
        """
        # Return a consistent mock image array for testing
        return np.ones((100, 100, 3)) * 128  # Gray image


def create_mock_plot_image(plot_class_name):
    """
    Create a mock plot object that simulates image generation.
    
    Args:
        plot_class_name (str): Name of the plot class to mock
    
    Returns:
        Mock: Mock plot object with save method
    """
    mock_plot = Mock()
    
    # Mock the save method to do nothing (no file I/O)
    def mock_save(path, **kwargs):
        # Simulate saving by just returning True
        return True
    
    mock_plot.save = mock_save
    mock_plot.close = Mock()
    mock_plot.ax = Mock()
    mock_plot.ax.get_title = Mock(return_value='Mock Plot Title')
    mock_plot.ax.get_xlabel = Mock(return_value='X Label')
    mock_plot.ax.get_ylabel = Mock(return_value='Y Label')
    
    # Add colorbar mock if needed
    mock_plot.cbar = Mock() if 'spatial' in plot_class_name.lower() else None
    
    return mock_plot


def mock_image_comparison_functions():
    """
    Create a context manager that mocks image comparison functions.
    
    Returns:
        Mock: Mock object containing mocked functions
    """
    mock_module = Mock()
    
    # Mock matplotlib.image.imread
    mock_module.imread = Mock(return_value=np.ones((100, 100, 3)) * 128)
    
    # Mock image comparison function
    mock_module.calculate_image_similarity = Mock(return_value={
        'pixel_match': 0.98,
        'structural_similarity': 0.98,
        'mean_pixel_difference': 0.02,
        'max_pixel_difference': 0.04,
        'identical_shape': True
    })
    
    return mock_module


def create_mock_plot_class(plot_class_name):
    """
    Create a mock plot class that simulates the actual plot classes.
    
    Args:
        plot_class_name (str): Name of the plot class to mock
    
    Returns:
        Mock: Mock plot class
    """
    mock_class = Mock()
    
    # Mock the plot method
    def mock_plot_method(data, *args, **kwargs):
        pass  # Do nothing in mock
    
    mock_class.plot = mock_plot_method
    mock_class.save = Mock(return_value=True)
    mock_class.close = Mock()
    mock_class.ax = Mock()
    mock_class.fig = Mock()
    
    # Set figure size
    mock_class.fig.get_size_inches = Mock(return_value=(8, 6))
    
    return mock_class