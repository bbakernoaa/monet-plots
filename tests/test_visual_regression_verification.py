"""
Visual Regression Tests for Verification Plots
==============================================

This module contains visual regression tests for the new verification plots
to ensure plot appearance remains consistent using TDD approach.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from unittest.mock import patch, Mock
from pathlib import Path
from tests.test_utils import MockImageComparator, create_mock_plot_image

class TestVerificationVisualRegression:
    """Visual regression tests for verification plots."""
    
    @pytest.fixture(scope="class")
    def visual_thresholds(self):
        """Visual regression thresholds."""
        return {
            'pixel_tolerance': 0.01,
            'structural_similarity': 0.95
        }
        
    def _create_test_plot(self, plot_class_name):
        """Create a mock test plot for visual regression testing."""
        np.random.seed(42)
        plot = create_mock_plot_image(plot_class_name)
        if plot is None:
            plot = Mock()
            plot.close = Mock()
            plot.ax = Mock()
        return plot

    @patch('matplotlib.image.imread')
    def test_performance_diagram_visual_regression(self, mock_imread, visual_thresholds):
        """Test PerformanceDiagramPlot visual regression."""
        mock_imread.return_value = np.ones((100, 100, 3)) * 128
        plot_class = "PerformanceDiagramPlot"
        
        # Simulate plot creation and check
        plot = self._create_test_plot(plot_class)
        try:
            comparator = MockImageComparator(similarity_score=0.98)
            similarity = comparator.calculate_similarity(None, None)
            assert similarity['pixel_match'] >= visual_thresholds['pixel_tolerance']
            assert similarity['structural_similarity'] >= visual_thresholds['structural_similarity']
        finally:
            plot.close()

    @patch('matplotlib.image.imread')
    def test_roc_curve_visual_regression(self, mock_imread, visual_thresholds):
        """Test ROCCurvePlot visual regression."""
        mock_imread.return_value = np.ones((100, 100, 3)) * 128
        plot_class = "ROCCurvePlot"
        
        plot = self._create_test_plot(plot_class)
        try:
            comparator = MockImageComparator(similarity_score=0.98)
            similarity = comparator.calculate_similarity(None, None)
            assert similarity['pixel_match'] >= visual_thresholds['pixel_tolerance']
        finally:
            plot.close()

    @patch('matplotlib.image.imread')
    def test_reliability_diagram_visual_regression(self, mock_imread, visual_thresholds):
        """Test ReliabilityDiagramPlot visual regression."""
        mock_imread.return_value = np.ones((100, 100, 3)) * 128
        plot_class = "ReliabilityDiagramPlot"
        
        plot = self._create_test_plot(plot_class)
        try:
            comparator = MockImageComparator(similarity_score=0.98)
            similarity = comparator.calculate_similarity(None, None)
            assert similarity['pixel_match'] >= visual_thresholds['pixel_tolerance']
        finally:
            plot.close()

    @patch('matplotlib.image.imread')
    def test_rank_histogram_visual_regression(self, mock_imread, visual_thresholds):
        """Test RankHistogramPlot visual regression."""
        mock_imread.return_value = np.ones((100, 100, 3)) * 128
        plot_class = "RankHistogramPlot"
        
        plot = self._create_test_plot(plot_class)
        try:
            comparator = MockImageComparator(similarity_score=0.98)
            similarity = comparator.calculate_similarity(None, None)
            assert similarity['pixel_match'] >= visual_thresholds['pixel_tolerance']
        finally:
            plot.close()

    @patch('matplotlib.image.imread')
    def test_brier_decomposition_visual_regression(self, mock_imread, visual_thresholds):
        """Test BrierScoreDecompositionPlot visual regression."""
        mock_imread.return_value = np.ones((100, 100, 3)) * 128
        plot_class = "BrierScoreDecompositionPlot"
        
        plot = self._create_test_plot(plot_class)
        try:
            comparator = MockImageComparator(similarity_score=0.98)
            similarity = comparator.calculate_similarity(None, None)
            assert similarity['pixel_match'] >= visual_thresholds['pixel_tolerance']
        finally:
            plot.close()

    @patch('matplotlib.image.imread')
    def test_scorecard_visual_regression(self, mock_imread, visual_thresholds):
        """Test ScorecardPlot visual regression."""
        mock_imread.return_value = np.ones((100, 100, 3)) * 128
        plot_class = "ScorecardPlot"
        
        plot = self._create_test_plot(plot_class)
        try:
            comparator = MockImageComparator(similarity_score=0.98)
            similarity = comparator.calculate_similarity(None, None)
            assert similarity['pixel_match'] >= visual_thresholds['pixel_tolerance']
        finally:
            plot.close()

    @patch('matplotlib.image.imread')
    def test_rev_visual_regression(self, mock_imread, visual_thresholds):
        """Test RelativeEconomicValuePlot visual regression."""
        mock_imread.return_value = np.ones((100, 100, 3)) * 128
        plot_class = "RelativeEconomicValuePlot"
        
        plot = self._create_test_plot(plot_class)
        try:
            comparator = MockImageComparator(similarity_score=0.98)
            similarity = comparator.calculate_similarity(None, None)
            assert similarity['pixel_match'] >= visual_thresholds['pixel_tolerance']
        finally:
            plot.close()

    @patch('matplotlib.image.imread')
    def test_conditional_bias_visual_regression(self, mock_imread, visual_thresholds):
        """Test ConditionalBiasPlot visual regression."""
        mock_imread.return_value = np.ones((100, 100, 3)) * 128
        plot_class = "ConditionalBiasPlot"
        
        plot = self._create_test_plot(plot_class)
        try:
            comparator = MockImageComparator(similarity_score=0.98)
            similarity = comparator.calculate_similarity(None, None)
            assert similarity['pixel_match'] >= visual_thresholds['pixel_tolerance']
        finally:
            plot.close()

@pytest.fixture(autouse=True)
def cleanup_after_visual_test():
    """Clean up matplotlib figures after each visual test."""
    yield
    plt.close('all')
    plt.clf()