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


def test_draw_map_docstring_example(clear_figures):
    """Tests the example from the SpatialPlot.draw_map docstring.

    This test validates that the function returns a valid matplotlib Axes
    object and that map features (like states) are drawn.
    """
    # --- The Logic (from docstring example) ---
    ax = SpatialPlot.draw_map(states=True, extent=[-125, -70, 25, 50])

    # --- The Proof (Validation) ---
    assert isinstance(ax, Axes), "The return type must be a matplotlib Axes object."

    # An empty map might have 1 collection (the spine). Adding states should
    # result in more collections being added.
    assert len(ax.collections) > 1, "Expected cartopy features to be drawn."


def test_spatial_track_docstring_example(clear_figures):
    """Tests the example from the SpatialTrack.plot docstring.

    This test validates that the plot method returns a PathCollection, which
    is the artist type for a scatter plot.
    """
    # --- The Logic (from docstring example) ---
    import numpy as np
    import xarray as xr
    from monet_plots.plots.spatial import SpatialTrack
    from matplotlib.collections import PathCollection

    # 1. Create a sample xarray.DataArray
    time = np.arange(20)
    lon = np.linspace(-120, -80, 20)
    lat = np.linspace(30, 45, 20)
    concentration = np.linspace(0, 100, 20)
    da = xr.DataArray(
        concentration,
        dims=["time"],
        coords={"time": time, "lon": ("time", lon), "lat": ("time", lat)},
        name="O3_concentration",
        attrs={"units": "ppb"},
    )

    # 2. Create and render the plot
    track_plot = SpatialTrack(data=da, states=True)
    sc = track_plot.plot(cmap="viridis")

    # --- The Proof (Validation) ---
    assert isinstance(sc, PathCollection), "The return type must be a PathCollection."
