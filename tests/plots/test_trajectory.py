import numpy as np
from monet_plots.plots import TrajectoryPlot

import pandas as pd


def test_TrajectoryPlot():
    lon = np.linspace(-120, -80, 10)
    lat = np.linspace(30, 40, 10)
    data = np.random.rand(10)
    time = pd.to_datetime(np.arange(10), unit="D")
    ts_data = np.random.rand(10)
    df = pd.DataFrame({"time": time, "value": ts_data})
    df.variable = "value"
    plot = TrajectoryPlot(lon, lat, data, df, "value")
    plot.plot()


def test_trajectory_axes_count():
    lon = np.linspace(-120, -80, 10)
    lat = np.linspace(30, 40, 10)
    data = np.random.rand(10)
    time = pd.date_range("2023-01-01", periods=10)
    ts_data = np.random.rand(10)

    plot = TrajectoryPlot(lon, lat, data, time, ts_data)
    plot.plot()

    # Map axes + Timeseries axes = 2
    assert len(plot.fig.axes) == 2


def test_spatial_track_auto_extent():
    from monet_plots.plots.spatial import SpatialTrack
    import xarray as xr

    lon = np.array([-10, 10])
    lat = np.array([-5, 5])
    data = np.array([0, 1])
    da = xr.DataArray(
        data, coords={"lon": ("time", lon), "lat": ("time", lat)}, dims="time"
    )

    plot = SpatialTrack(da)
    plot.plot()

    # Extent should be roughly [-10, 10, -5, 5] plus buffer
    extent = plot.ax.get_extent()
    # Check that it is localized and not global (-180, 180 etc)
    assert -180 < extent[0] < -10
    assert 10 < extent[1] < 180
    assert -90 < extent[2] < -5
    assert 5 < extent[3] < 90
