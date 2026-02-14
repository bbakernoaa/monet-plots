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
from monet_plots.plots.taylor_diagram import TaylorDiagramPlot


@pytest.mark.skipif(da is None, reason="dask not installed")
def test_taylor_diagram_lazy_parity():
    """Verify TaylorDiagramPlot handles lazy data and matches eager results."""
    # Create test data
    x = np.linspace(0, 10, 100)
    obs = np.sin(x)
    mod1 = obs + 0.1 * np.random.randn(100)
    mod2 = 0.8 * obs + 0.2 * np.random.randn(100)

    ds_eager = xr.Dataset(
        {
            "obs": (["time"], obs),
            "model1": (["time"], mod1),
            "model2": (["time"], mod2),
        },
        coords={"time": x},
    )
    ds_lazy = ds_eager.chunk({"time": 50})

    # Track A: Eager
    plot_eager = TaylorDiagramPlot(ds_eager, col1="obs", col2=["model1", "model2"])
    dia_eager = plot_eager.plot()
    assert dia_eager is not None
    # Reference point + 2 models
    assert len(dia_eager.samplePoints) == 3
    plt.close(dia_eager._ax.figure)

    # Track B: Lazy
    plot_lazy = TaylorDiagramPlot(ds_lazy, col1="obs", col2=["model1", "model2"])
    # History check (Initialized)
    assert "Initialized TaylorDiagramPlot" in plot_lazy.df.attrs["history"]

    dia_lazy = plot_lazy.plot()
    assert dia_lazy is not None
    assert len(dia_lazy.samplePoints) == 3

    # Check history (Generated)
    assert "Generated TaylorDiagramPlot" in plot_lazy.df.attrs["history"]

    # Parity check: compare coordinates of sample points (theta, r)
    # The first sample is always the reference point at (0, obs_std)
    for i in range(3):
        # dia_lazy.samplePoints[i] is a Line2D object
        theta_eager, r_eager = dia_eager.samplePoints[i].get_data()
        theta_lazy, r_lazy = dia_lazy.samplePoints[i].get_data()
        np.testing.assert_allclose(theta_eager, theta_lazy)
        np.testing.assert_allclose(r_eager, r_lazy)

        # Style check: ensure models (indices 1 and 2) are symbols, not lines
        if i > 0:
            assert dia_lazy.samplePoints[i].get_linestyle() in ["", "None"]
            assert dia_lazy.samplePoints[i].get_marker() not in ["", "None", None]

    plt.close(dia_lazy._ax.figure)


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
