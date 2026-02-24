import numpy as np
import pandas as pd
import xarray as xr
import pytest
import dask.array as da
from monet_plots.plots import TrajectoryPlot


def test_trajectory_lazy_xarray():
    """Test TrajectoryPlot with Dask-backed xarray objects."""
    n = 100
    lon = xr.DataArray(
        da.from_array(np.linspace(-120, -70, n), chunks=50), dims="time", name="lon"
    )
    lat = xr.DataArray(
        da.from_array(np.linspace(25, 50, n), chunks=50), dims="time", name="lat"
    )
    data = xr.DataArray(
        da.from_array(np.random.rand(n), chunks=50), dims="time", name="concentration"
    )

    # Time series data in the same or different object
    times = pd.date_range("2023-01-01", periods=n, freq="h")
    ts_data = xr.DataArray(
        da.from_array(np.random.rand(n), chunks=50),
        coords={"time": times},
        dims="time",
        name="obs",
    )

    # 1. Test initialization with separate xarray objects
    plot = TrajectoryPlot(
        longitude=lon, latitude=lat, data=data, time=times, ts_data=ts_data
    )

    # Verify that data remains lazy (Dask-backed)
    # After normalize_data, it should still be a DataArray with Dask backend
    assert isinstance(plot.data.data, da.Array)

    # 2. Test plot()
    axs = plot.plot()
    assert len(axs) == 2

    # 3. Test provenance
    assert "Initialized TrajectoryPlot (Map)" in plot.data.attrs["history"]


def test_trajectory_dataset_input():
    """Test TrajectoryPlot where time and ts_data are in the same Dataset."""
    n = 10
    lon = np.linspace(-120, -70, n)
    lat = np.linspace(25, 50, n)
    data = np.random.rand(n)

    times = pd.date_range("2023-01-01", periods=n)
    ds = xr.Dataset({"obs": (("time"), np.random.rand(n))}, coords={"time": times})

    # Here time=ds and ts_data="obs" (column name)
    plot = TrajectoryPlot(lon, lat, data, ds, "obs")
    axs = plot.plot()
    assert len(axs) == 2


def test_trajectory_hvplot():
    """Test the hvplot() method of TrajectoryPlot."""
    pytest.importorskip("hvplot")
    pytest.importorskip("holoviews")

    n = 20
    lon = np.linspace(-120, -70, n)
    lat = np.linspace(25, 50, n)
    data = np.random.rand(n)
    time = pd.date_range("2023-01-01", periods=n)
    ts_data = np.random.rand(n)

    plot = TrajectoryPlot(lon, lat, data, time, ts_data)
    hv_obj = plot.hvplot()

    import holoviews as hv

    assert isinstance(hv_obj, hv.Layout)
    assert len(hv_obj) == 2
