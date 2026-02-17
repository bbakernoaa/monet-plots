import numpy as np
import pytest
import xarray as xr
from monet_plots.plots.spatial import SpatialPlot

try:
    import dask.array as da
except ImportError:
    da = None


def test_identify_coords():
    """Verify coordinate identification via CF names and attributes."""
    data = np.random.rand(10, 10)

    # Case 1: Standard names
    da1 = xr.DataArray(
        data,
        coords={"longitude": np.arange(10), "latitude": np.arange(10)},
        dims=["latitude", "longitude"],
    )
    lon, lat = SpatialPlot._identify_coords(da1)
    assert lon == "longitude"
    assert lat == "latitude"

    # Case 2: Short names with units
    da2 = xr.DataArray(
        data, coords={"ln": np.arange(10), "lt": np.arange(10)}, dims=["lt", "ln"]
    )
    da2.ln.attrs["units"] = "degrees_east"
    da2.lt.attrs["units"] = "degrees_north"
    lon, lat = SpatialPlot._identify_coords(da2)
    assert lon == "ln"
    assert lat == "lt"


def test_ensure_monotonic():
    """Verify that latitude is sorted to be increasing."""
    lat = np.array([40, 39, 38])
    lon = np.array([-100, -99, -98])
    data = np.random.rand(3, 3)

    da_input = xr.DataArray(data, coords={"lat": lat, "lon": lon}, dims=["lat", "lon"])

    assert da_input.lat.values[0] > da_input.lat.values[-1]

    da_fixed = SpatialPlot._ensure_monotonic(da_input)

    assert da_fixed.lat.values[0] < da_fixed.lat.values[-1]
    assert np.all(da_fixed.lat.values == np.sort(lat))


@pytest.mark.skipif(da is None, reason="dask not installed")
def test_get_extent_from_data_lazy():
    """Verify extent computation for Dask-backed objects."""
    lat = np.linspace(30, 40, 10)
    lon = np.linspace(-100, -90, 10)
    data = np.random.rand(10, 10)

    da_lazy = xr.DataArray(
        data, coords={"lat": lat, "lon": lon}, dims=["lat", "lon"]
    ).chunk({"lat": 5, "lon": 5})

    extent = SpatialPlot._get_extent_from_data(da_lazy)

    assert extent == [-100.0, -90.0, 30.0, 40.0]
    assert isinstance(extent[0], float)


if __name__ == "__main__":
    pytest.main([__file__])
