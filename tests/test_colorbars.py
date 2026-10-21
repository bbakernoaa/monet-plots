import matplotlib.pyplot as plt
import numpy as np
import pytest

from monet_plots.colorbars import cmap_discretize, colorbar_index


@pytest.fixture
def clear_figures():
    """Clear all existing figures before and after a test."""
    plt.close("all")
    yield
    plt.close("all")


def test_colorbar_index_creates_colorbar(clear_figures):
    """Test that colorbar_index creates a colorbar."""
    fig, ax = plt.subplots()
    ax.imshow(np.random.rand(10, 10))
    cbar, cmap = colorbar_index(ncolors=10, cmap="viridis", ax=ax)
    assert cbar is not None
    assert cmap is not None


def test_cmap_discretize_creates_cmap():
    """Test that cmap_discretize creates a colormap."""
    cmap = cmap_discretize("viridis", 10)
    assert cmap is not None
