import numpy as np
import pytest
import xarray as xr
from monet_plots.plots.windrose import Windrose

def test_windrose_numpy():
    """Test Windrose with numpy arrays."""
    wd = np.random.uniform(0, 360, 100)
    ws = np.random.uniform(0, 20, 100)

    wr = Windrose(wd=wd, ws=ws)
    ax = wr.plot(bins=8, rose_bins=4)

    assert ax is not None
    # Check that we have bars (stacked, so 8 directions * 4 speed bins = 32 patches if flattened,
    # but they are added in a loop over speeds)
    # Actually, it's 8 bars per speed bin. 4 speed bins -> 32 bars.
    assert len(ax.patches) == 32
    wr.close()

def test_windrose_lazy():
    """Test Windrose with lazy dask-backed xarray objects."""
    dask = pytest.importorskip("dask")

    wd_arr = np.random.uniform(0, 360, 100)
    ws_arr = np.random.uniform(0, 20, 100)

    wd = xr.DataArray(wd_arr, dims="time").chunk({"time": 50})
    ws = xr.DataArray(ws_arr, dims="time").chunk({"time": 50})

    wr = Windrose(wd=wd, ws=ws)
    ax = wr.plot(bins=8, rose_bins=4)

    assert ax is not None
    assert len(ax.patches) == 32

    # Check history (provenance)
    assert "Plotted Windrose" in wr.wd.attrs.get("history", "")
    assert "Plotted Windrose" in wr.ws.attrs.get("history", "")

    wr.close()

def test_windrose_consistency():
    """Verify that eager and lazy paths yield the same underlying histogram."""
    dask = pytest.importorskip("dask")

    # Use fixed data for consistency check
    wd_arr = np.linspace(0, 360, 100, endpoint=False)
    ws_arr = np.linspace(0, 20, 100)

    # Eager path
    wr_eager = Windrose(wd=wd_arr, ws=ws_arr)
    ax_eager = wr_eager.plot(bins=8, rose_bins=4)
    eager_heights = [p.get_height() for p in ax_eager.patches]
    wr_eager.close()

    # Lazy path
    wd_lazy = xr.DataArray(wd_arr, dims="time").chunk({"time": 50})
    ws_lazy = xr.DataArray(ws_arr, dims="time").chunk({"time": 50})
    wr_lazy = Windrose(wd=wd_lazy, ws=ws_lazy)
    ax_lazy = wr_lazy.plot(bins=8, rose_bins=4)
    lazy_heights = [p.get_height() for p in ax_lazy.patches]
    wr_lazy.close()

    np.testing.assert_allclose(eager_heights, lazy_heights)

def test_windrose_hvplot():
    """Test the hvplot method (Track B)."""
    hvplot = pytest.importorskip("hvplot")

    wd = np.random.uniform(0, 360, 100)
    ws = np.random.uniform(0, 20, 100)

    wr = Windrose(wd=wd, ws=ws)
    plot = wr.hvplot(bins=8, rose_bins=4)

    assert plot is not None
    wr.close()
