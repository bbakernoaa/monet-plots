import numpy as np
import pytest
import xarray as xr
from monet_plots.plots.spatial import SpatialPlot

try:
    import dask.array as da

    HAS_DASK = True
except ImportError:
    HAS_DASK = False


def test_get_extent_from_data_eager():
    # Create eager data
    lon = np.linspace(-120, -70, 10)
    lat = np.linspace(25, 50, 10)
    data = xr.DataArray(
        np.random.rand(10, 10),
        coords={"lat": lat, "lon": lon},
        dims=("lat", "lon"),
        name="test",
    )
    # Set units to help identification
    data.lat.attrs["units"] = "degrees_north"
    data.lon.attrs["units"] = "degrees_east"

    plot = SpatialPlot()
    extent = plot._get_extent_from_data(data)

    assert extent == [-120.0, -70.0, 25.0, 50.0]


@pytest.mark.skipif(not HAS_DASK, reason="dask not installed")
def test_get_extent_from_data_lazy():
    # Create lazy data
    lon = np.linspace(-120, -70, 10)
    lat = np.linspace(25, 50, 10)
    data = xr.DataArray(
        da.random.random((10, 10), chunks=(5, 5)),
        coords={"lat": lat, "lon": lon},
        dims=("lat", "lon"),
        name="test",
    )
    # Set units to help identification
    data.lat.attrs["units"] = "degrees_north"
    data.lon.attrs["units"] = "degrees_east"

    plot = SpatialPlot()
    extent = plot._get_extent_from_data(data)

    assert extent == [-120.0, -70.0, 25.0, 50.0]


def test_get_extent_from_data_with_buffer():
    lon = np.linspace(-100, -80, 11)
    lat = np.linspace(30, 40, 11)
    data = xr.DataArray(
        np.random.rand(11, 11),
        coords={"lat": lat, "lon": lon},
        dims=("lat", "lon"),
        name="test",
    )
    data.lat.attrs["units"] = "degrees_north"
    data.lon.attrs["units"] = "degrees_east"

    plot = SpatialPlot()
    extent = plot._get_extent_from_data(data, buffer=0.1)

    # Range lon: 20, 10% = 2 -> [-102, -78]
    # Range lat: 10, 10% = 1 -> [29, 41]
    assert pytest.approx(extent) == [-102.0, -78.0, 29.0, 41.0]


def test_get_extent_from_dataset():
    lon = np.linspace(-120, -70, 10)
    lat = np.linspace(25, 50, 10)
    ds = xr.Dataset(
        {"var1": (("lat", "lon"), np.random.rand(10, 10))},
        coords={"lat": lat, "lon": lon},
    )
    ds.lat.attrs["units"] = "degrees_north"
    ds.lon.attrs["units"] = "degrees_east"

    plot = SpatialPlot()
    extent = plot._get_extent_from_data(ds)

    assert extent == [-120.0, -70.0, 25.0, 50.0]
