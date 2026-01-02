# tests/plots/test_spatial_validation.py
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pytest
from monet_plots.plots.spatial import SpatialPlot


@pytest.fixture
def clear_figures():
    """Clear all existing figures before and after a test."""
    plt.close("all")
    yield
    plt.close("all")


@pytest.mark.skip(reason="Depends on cartopy asset download, which fails in CI.")
def test_create_map_docstring_example(clear_figures):
    """Tests the example from the SpatialPlot.create_map docstring.

    This test validates that the function returns a valid SpatialPlot object
    and that map features (like states) are drawn on its Axes.
    """
    # --- The Logic (from docstring example) ---
    # Use 110m resolution as it's more likely to be cached in test environments.
    plot = SpatialPlot.create_map(
        states=True, extent=[-125, -70, 25, 50], resolution="110m"
    )

    # --- The Proof (Validation) ---
    assert isinstance(plot, SpatialPlot), "The return type must be a SpatialPlot object."
    assert isinstance(plot.ax, Axes), "The plot.ax attribute must be a matplotlib Axes object."

    # Force a draw to ensure collections are updated before the assertion.
    plot.fig.canvas.draw()

    # An empty map might have 1 collection (the spine). Adding states should
    # result in more collections being added.
    assert len(plot.ax.collections) > 1, "Expected cartopy features to be drawn."


def test_spatial_track_docstring_example(clear_figures):
    """Tests the example from the SpatialTrack.plot docstring.

    This test validates that the plot method returns a PathCollection, which
    is the artist type for a scatter plot.
    """
    # --- The Logic (from docstring example) ---
    import numpy as np
    from monet_plots.plots.spatial import SpatialTrack
    from matplotlib.collections import PathCollection

    lon = np.linspace(-120, -80, 20)
    lat = np.linspace(30, 45, 20)
    data = np.linspace(0, 100, 20)
    track_plot = SpatialTrack(lon, lat, data, states=True)
    sc = track_plot.plot(cmap="viridis")

    # --- The Proof (Validation) ---
    assert isinstance(sc, PathCollection), "The return type must be a PathCollection."
