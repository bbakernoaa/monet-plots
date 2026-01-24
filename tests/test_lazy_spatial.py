# tests/test_lazy_spatial.py
import pytest

# Check if dask and dask.array are installed
da = pytest.importorskip("dask.array")
xr = pytest.importorskip("xarray")

import numpy as np
import pandas as pd
from matplotlib.artist import Artist

from monet_plots.plots.spatial import SpatialTrack


def test_spatial_track_plot_dask_awareness():
    """Test that SpatialTrack.plot() preserves Dask laziness.

    This test verifies that when plotting a Dask-backed xarray.DataArray,
    the plot method does not trigger an immediate computation of the array,
    adhering to the "Lazy by Default" principle of the Aero Protocol.
    """
    # 1. The Logic (Setup): Create a Dask-backed xarray.DataArray
    time = pd.date_range("2024-07-25", periods=100, freq="h")
    lat = da.from_array(np.linspace(30, 40, 100), chunks=(50,))
    lon = da.from_array(np.linspace(-100, -80, 100), chunks=(50,))
    altitude = da.from_array(np.linspace(1000, 5000, 100), chunks=(50,))

    data = xr.DataArray(
        altitude,
        coords={"time": time, "lat": ("time", lat), "lon": ("time", lon)},
        dims=["time"],
        name="altitude",
    )

    # Confirm that the data is indeed a Dask array
    assert hasattr(data.data, "dask")

    # 2. The UI (Execution): Instantiate and plot
    track_plot = SpatialTrack(data)
    scatter_artist = track_plot.plot()

    # 3. The Proof (Validation):
    # a) Check that the returned object is a Matplotlib artist
    assert isinstance(scatter_artist, Artist), "The plot method should return a Matplotlib artist."

    # b) The core assertion: Verify that the Dask array has NOT been computed.
    # The `_in_memory` property is False if the data is still lazy.
    assert not data.variable._in_memory, "The Dask array was computed eagerly, violating laziness."

    # c) Check history for provenance
    assert "Plotted with monet-plots.SpatialTrack" in data.attrs["history"]
