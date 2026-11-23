import numpy as np
import pytest
import xarray as xr

try:
    import dask.array as da
except ImportError:
    da = None
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

from monet_plots.plots.scatter import ScatterPlot
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


@pytest.mark.skipif(da is None, reason="dask not installed")
def test_scatter_plot_eager_vs_lazy():
    """Verify ScatterPlot handles both numpy and dask backends."""
    # Create test data
    x = np.linspace(0, 10, 100)
    y = 2 * x + 1 + np.random.randn(100)

    ds_eager = xr.Dataset(
        {"x": (["index"], x), "y": (["index"], y)}, coords={"index": np.arange(100)}
    )
    ds_lazy = ds_eager.chunk({"index": 50})

    # Track A: Eager
    plot_eager = ScatterPlot(ds_eager, x="x", y="y")
    assert isinstance(plot_eager.data, xr.Dataset)

    ax_eager = plot_eager.plot()
    assert ax_eager is not None
    assert len(ax_eager.collections) > 0  # Scatter points
    assert len(ax_eager.lines) > 0  # Regression line
    plt.close(ax_eager.figure)

    # Track B: Lazy
    plot_lazy = ScatterPlot(ds_lazy, x="x", y="y")
    assert hasattr(plot_lazy.data.x.data, "dask")

    ax_lazy = plot_lazy.plot()
    assert ax_lazy is not None
    assert len(ax_lazy.collections) > 0
    assert len(ax_lazy.lines) > 0
    plt.close(ax_lazy.figure)


@pytest.mark.skipif(da is None, reason="dask not installed")
def test_scatter_plot_geospatial():
    """Verify ScatterPlot works on GeoAxes with mandatory transform."""
    x = np.linspace(-120, -70, 10)
    y = np.linspace(25, 50, 10)
    ds = xr.Dataset({"x": (["index"], x), "y": (["index"], y)})

    # Initialize with projection to create GeoAxes
    plot = ScatterPlot(ds, x="x", y="y", subplot_kw={"projection": ccrs.PlateCarree()})
    assert hasattr(plot.ax, "projection")

    # This should now NOT fail and apply the transform
    ax = plot.plot()
    assert ax is not None
    plt.close(ax.figure)


if __name__ == "__main__":
    pytest.main([__file__])


def test_timeseries_materializes_reduced_mean_and_spread_once():
    from dask.callbacks import Callback

    from monet_plots.plots.timeseries import TimeSeriesPlot

    data = xr.DataArray(
        np.arange(24, dtype=float).reshape(8, 3),
        dims=("time", "station"),
        coords={"time": np.arange(8)},
        name="temperature",
    )
    data[3] = np.nan
    eager_plot = TimeSeriesPlot(data)
    lazy_plot = TimeSeriesPlot(data.chunk({"time": 3, "station": 1}))
    starts = []
    try:
        eager = eager_plot.plot()
        with Callback(start=lambda graph: starts.append(len(graph))):
            lazy = lazy_plot.plot()
        assert len(starts) == 1
        np.testing.assert_allclose(
            lazy.lines[0].get_ydata(), eager.lines[0].get_ydata()
        )
        for lazy_path, eager_path in zip(
            lazy.collections[0].get_paths(),
            eager.collections[0].get_paths(),
            strict=True,
        ):
            np.testing.assert_allclose(lazy_path.vertices, eager_path.vertices)
    finally:
        eager_plot.close()
        lazy_plot.close()
