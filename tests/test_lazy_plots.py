import numpy as np
import pytest
import xarray as xr

try:
    import dask.array as da
except ImportError:
    da = None
from monet_plots.plots.spatial_imshow import SpatialImshowPlot
from monet_plots.plots.performance_diagram import PerformanceDiagramPlot
from monet_plots.plots.soccer import SoccerPlot


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


@pytest.mark.skipif(da is None, reason="dask not installed")
def test_performance_diagram_lazy():
    """Verify PerformanceDiagramPlot handles lazy dask inputs."""
    shape = (100,)
    hits = xr.DataArray(
        da.from_array(np.random.randint(1, 10, shape), chunks=20),
        dims=["x"],
        name="hits",
    )
    misses = xr.DataArray(
        da.from_array(np.random.randint(1, 10, shape), chunks=20),
        dims=["x"],
        name="misses",
    )
    fa = xr.DataArray(
        da.from_array(np.random.randint(1, 10, shape), chunks=20), dims=["x"], name="fa"
    )
    cn = xr.DataArray(
        da.from_array(np.random.randint(1, 10, shape), chunks=20), dims=["x"], name="cn"
    )

    ds = xr.Dataset({"hits": hits, "misses": misses, "fa": fa, "cn": cn})

    plot = PerformanceDiagramPlot()
    # This should stay lazy
    df_plot = plot._prepare_data(ds, "sr", "pod", ["hits", "misses", "fa", "cn"])

    assert hasattr(df_plot.sr.data, "chunks")
    assert hasattr(df_plot.pod.data, "chunks")

    # Plotting should work (triggers computation)
    ax = plot.plot(ds, counts_cols=["hits", "misses", "fa", "cn"])
    assert ax is not None
    plot.close()


@pytest.mark.skipif(da is None, reason="dask not installed")
def test_soccer_plot_lazy():
    """Verify SoccerPlot handles lazy dask inputs."""
    shape = (100,)
    obs = xr.DataArray(
        da.from_array(np.random.rand(*shape) + 1.0, chunks=20), dims=["x"], name="obs"
    )
    mod = xr.DataArray(
        da.from_array(np.random.rand(*shape) + 1.0, chunks=20), dims=["x"], name="mod"
    )

    ds = xr.Dataset({"obs": obs, "mod": mod})

    plot = SoccerPlot(ds, obs_col="obs", mod_col="mod", metric="fractional")

    assert hasattr(plot.bias_data.data, "chunks")
    assert hasattr(plot.error_data.data, "chunks")

    ax = plot.plot()
    assert ax is not None
    plot.close()


if __name__ == "__main__":
    pytest.main([__file__])
