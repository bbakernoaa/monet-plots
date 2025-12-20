from matplotlib.quiver import Barbs
import pytest
import matplotlib.pyplot as plt
import numpy as np
from monet_plots.plots.profile import (
    ProfilePlot,
    VerticalSlice,
    StickPlot,
    VerticalBoxPlot,
)


@pytest.fixture
def clear_figures():
    """Clear all existing figures before and after a test."""
    plt.close("all")
    yield
    plt.close("all")


@pytest.fixture
def sample_data_line():
    """Create sample data for a line plot."""
    return {
        "x": np.linspace(0, 10, 100),
        "y": np.linspace(0, 10, 100) + np.random.rand(100),
    }


@pytest.fixture
def sample_data_contour():
    """Create sample data for a contour plot."""
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 10, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X) * np.cos(Y)
    return {"x": X, "y": Y, "z": Z}


def test_profile_plot_line_creates_plot(clear_figures, sample_data_line):
    """Test that ProfilePlot creates a line plot."""
    plot = ProfilePlot(**sample_data_line)
    plot.plot()
    assert plot.ax is not None
    assert len(plot.ax.lines) > 0


def test_profile_plot_contour_creates_plot(clear_figures, sample_data_contour):
    """Test that ProfilePlot creates a contour plot."""
    plot = ProfilePlot(**sample_data_contour)
    plot.plot()
    assert plot.ax is not None
    assert len(plot.ax.collections) > 0


def test_profile_plot_alt_adjust(clear_figures, sample_data_line):
    """Test that ProfilePlot adjusts altitude correctly."""
    alt_adjust = 5.0
    original_y = sample_data_line["y"].copy()
    plot = ProfilePlot(**sample_data_line, alt_adjust=alt_adjust)
    assert np.allclose(plot.y, original_y - alt_adjust)


def test_VerticalBoxPlot_plot(clear_figures):
    """Test that VerticalBoxPlot creates a plot."""
    data = np.random.rand(100)
    y = np.linspace(0, 10, 100)
    thresholds = [0, 2, 4, 6, 8, 10]
    plot = VerticalBoxPlot(data, y, thresholds)
    result = plot.plot()
    assert plot.ax is not None
    assert len(result["boxes"]) > 0


def test_VerticalSlice_plot(clear_figures, sample_data_contour):
    """Test that VerticalSlice creates a contour plot."""
    plot = VerticalSlice(**sample_data_contour)
    plot.plot()
    assert plot.ax is not None
    assert len(plot.ax.collections) > 0


def test_StickPlot_plot(clear_figures):
    """Test that StickPlot creates a plot."""
    u = np.random.rand(10) * 10
    v = np.random.rand(10) * 10
    y = np.arange(10)
    plot = StickPlot(u, v, y)
    result = plot.plot()
    assert plot.ax is not None
    assert isinstance(result, Barbs)
