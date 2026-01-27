import numpy as np
import pytest
import xarray as xr

try:
    import dask.array as da
except ImportError:
    da = None
from monet_plots.plots.spatial_imshow import SpatialImshowPlot


@pytest.mark.skipif(da is None, reason="dask not installed")
def test_spatial_imshow_eager_vs_lazy():
    """Verify SpatialImshowPlot handles both numpy and dask backends."""
    # Create test data
    data = np.random.rand(10, 10)
    lat = np.linspace(30, 40, 10)
    lon = np.linspace(-100, -90, 10)

    da_eager = xr.DataArray(
        data, coords={"lat": lat, "lon": lon}, dims=["lat", "lon"], name="test"
    )
    da_lazy = da_eager.chunk({"lat": 5, "lon": 5})

    # Track A: Eager
    plot_eager = SpatialImshowPlot(da_eager)
    assert isinstance(plot_eager.modelvar, xr.DataArray)

    # Check if we can still call plot
    cbar_eager = plot_eager.plot()
    assert cbar_eager is not None
    plot_eager.close()

    # Track B: Lazy
    plot_lazy = SpatialImshowPlot(da_lazy)
    assert hasattr(plot_lazy.modelvar.data, "dask")

    cbar_lazy = plot_lazy.plot()
    assert cbar_lazy is not None
    plot_lazy.close()


if __name__ == "__main__":
    pytest.main([__file__])
