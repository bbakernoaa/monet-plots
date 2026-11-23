import numpy as np
import xarray as xr

from monet_plots.plots.spatial import SpatialTrack
from monet_plots.plots.spatial_bias_scatter import SpatialBiasScatterPlot
from monet_plots.plots.spatial_contour import SpatialContourPlot
from monet_plots.plots.spatial_imshow import SpatialImshowPlot


def create_mock_data(lazy=False):
    """Create mock xarray data for testing."""
    lon = np.linspace(-120, -70, 10)
    lat = np.linspace(25, 50, 10)
    data = np.random.rand(10, 10)

    da = xr.DataArray(
        data, coords={"lon": lon, "lat": lat}, dims=("lat", "lon"), name="test_var"
    )

    if lazy:
        da = da.chunk({"lat": 5, "lon": 5})

    return da


def test_imshow_laziness():
    """Verify SpatialImshowPlot handles lazy data without eager compute during init."""
    da = create_mock_data(lazy=True)

    # Initialize plot
    plot = SpatialImshowPlot(da)

    # Verify data is still lazy (has chunks)
    assert hasattr(plot.modelvar.data, "chunks")
    # Verify history was updated
    assert "Initialized monet-plots.SpatialImshowPlot" in plot.modelvar.attrs["history"]


def test_contour_laziness():
    """Verify SpatialContourPlot handles lazy data without eager compute during init."""
    da = create_mock_data(lazy=True)

    # Initialize plot
    plot = SpatialContourPlot(da, None)

    # Verify data is still lazy
    assert hasattr(plot.modelvar.data, "chunks")
    # Verify history was updated
    assert (
        "Initialized monet-plots.SpatialContourPlot" in plot.modelvar.attrs["history"]
    )


def test_track_laziness():
    """Verify SpatialTrack handles lazy data without eager compute during init."""
    da = create_mock_data(lazy=True)
    # Track needs a 1D path usually, but let's just use the coordinates
    track_da = da.isel(lon=0)  # 1D along lat

    # Initialize plot
    plot = SpatialTrack(track_da)

    # Verify data is still lazy
    assert hasattr(plot.data.data, "chunks")
    # Verify history was updated
    assert "Plotted with monet-plots.SpatialTrack" in plot.data.attrs["history"]


def test_spatial_bias_scatter_laziness():
    """Verify SpatialBiasScatterPlot handles lazy data without eager compute during init."""
    lon = np.linspace(-120, -70, 100)
    lat = np.linspace(25, 50, 100)
    obs = np.random.rand(100)
    mod = np.random.rand(100)

    ds = xr.Dataset(
        {
            "obs": (["sample"], obs),
            "mod": (["sample"], mod),
            "lat": (["sample"], lat),
            "lon": (["sample"], lon),
        }
    ).chunk({"sample": 50})

    # Initialize plot
    plot = SpatialBiasScatterPlot(ds, "obs", "mod")

    # Verify data is still lazy
    assert hasattr(plot.data.obs.data, "chunks")
    # Verify history was updated
    assert (
        "Initialized monet-plots.SpatialBiasScatterPlot" in plot.data.attrs["history"]
    )


def test_imshow_plot_parity():
    """Verify SpatialImshowPlot produces an axes when plotted with both eager and lazy data."""
    import matplotlib.pyplot as plt

    plt.switch_backend("Agg")

    eager_da = create_mock_data(lazy=False)
    lazy_da = create_mock_data(lazy=True)

    # Eager plot
    plot_eager = SpatialImshowPlot(eager_da)
    ax_eager = plot_eager.plot()
    assert ax_eager is not None

    # Lazy plot (triggers compute internally for imshow)
    plot_lazy = SpatialImshowPlot(lazy_da)
    ax_lazy = plot_lazy.plot()
    assert ax_lazy is not None

    plt.close("all")


def test_spatial_facet_grid_laziness():
    """Verify SpatialFacetGridPlot handles lazy data without eager compute."""
    import matplotlib.pyplot as plt
    from monet_plots.plots.facet_grid import SpatialFacetGridPlot

    plt.switch_backend("Agg")

    # Create 3D data (time, lat, lon)
    lon = np.linspace(-120, -70, 10)
    lat = np.linspace(25, 50, 10)
    time = np.arange(3)
    data = np.random.rand(3, 10, 10)

    da = xr.DataArray(
        data,
        coords={"time": time, "lon": lon, "lat": lat},
        dims=("time", "lat", "lon"),
        name="test_var",
    ).chunk({"time": 1, "lat": 5, "lon": 5})

    # Initialize facet grid
    fg = SpatialFacetGridPlot(da, col="time")

    # Verify data is still lazy
    assert hasattr(fg.data.data, "chunks")

    # Map a plotter (this should still be lazy in terms of the full array)
    fg.map_monet(SpatialImshowPlot)

    # Verify we have 3 axes
    _axes = fg.grid.axs if hasattr(fg.grid, "axs") else fg.grid.axes
    assert len(_axes.flatten()) >= 3

    plt.close("all")


def test_spatial_imshow_auto_facet():
    """Verify SpatialImshowPlot automatically redirects to FacetGrid for multiple facets."""
    import matplotlib.pyplot as plt

    plt.switch_backend("Agg")

    lon = np.linspace(-120, -70, 10)
    lat = np.linspace(25, 50, 10)
    time = np.arange(2)
    da = xr.DataArray(
        np.random.rand(2, 10, 10),
        coords={"time": time, "lon": lon, "lat": lat},
        dims=("time", "lat", "lon"),
        name="test_var",
    ).chunk({"time": 1})

    # This should return a SpatialFacetGridPlot instance due to __new__
    plot = SpatialImshowPlot(da, col="time")

    from monet_plots.plots.facet_grid import SpatialFacetGridPlot

    assert isinstance(plot, SpatialFacetGridPlot)
    assert plot.col == "time"

    plt.close("all")


def test_spatial_imshow_col_wrap():
    """Verify col_wrap works and triggers redirection."""
    import matplotlib.pyplot as plt

    plt.switch_backend("Agg")

    lon = np.linspace(-120, -70, 10)
    lat = np.linspace(25, 50, 10)
    time = np.arange(4)
    da = xr.DataArray(
        np.random.rand(4, 10, 10),
        coords={"time": time, "lon": lon, "lat": lat},
        dims=("time", "lat", "lon"),
        name="test_var",
    )

    # Trigger with col_wrap
    plot = SpatialImshowPlot(da, col="time", col_wrap=2)

    from monet_plots.plots.facet_grid import SpatialFacetGridPlot

    assert isinstance(plot, SpatialFacetGridPlot)
    assert plot.col == "time"
    assert plot.col_wrap == 2

    plt.close("all")


def test_contour_plot_parity():
    """Verify SpatialContourPlot produces an axes when plotted with both eager and lazy data."""
    import matplotlib.pyplot as plt

    plt.switch_backend("Agg")

    eager_da = create_mock_data(lazy=False)
    lazy_da = create_mock_data(lazy=True)

    # Eager plot
    plot_eager = SpatialContourPlot(eager_da, None)
    ax_eager = plot_eager.plot(levels=5)
    assert ax_eager is not None

    # Lazy plot
    plot_lazy = SpatialContourPlot(lazy_da, None)
    ax_lazy = plot_lazy.plot(levels=5)
    assert_lazy = ax_lazy is not None
    assert assert_lazy

    plt.close("all")


def test_spatial_bias_scatter_plot_parity():
    """Verify SpatialBiasScatterPlot produces an axes when plotted with both eager and lazy data."""
    import matplotlib.pyplot as plt

    plt.switch_backend("Agg")

    lon = np.linspace(-120, -70, 10)
    lat = np.linspace(25, 50, 10)
    obs = np.random.rand(10)
    mod = np.random.rand(10)

    ds = xr.Dataset(
        {
            "obs": (["sample"], obs),
            "mod": (["sample"], mod),
            "lat": (["sample"], lat),
            "lon": (["sample"], lon),
        }
    )

    lazy_ds = ds.chunk({"sample": 5})

    # Eager plot
    plot_eager = SpatialBiasScatterPlot(ds, "obs", "mod")
    ax_eager = plot_eager.plot()
    assert ax_eager is not None

    # Lazy plot
    plot_lazy = SpatialBiasScatterPlot(lazy_ds, "obs", "mod")
    ax_lazy = plot_lazy.plot()
    assert ax_lazy is not None

    plt.close("all")


def test_discrete_contour_uses_one_dask_pass_and_matches_eager():
    import matplotlib.pyplot as plt
    from dask.callbacks import Callback

    data = xr.DataArray(
        np.arange(48, dtype=float).reshape(6, 8),
        dims=("lat", "lon"),
        coords={"lat": np.linspace(30, 40, 6), "lon": np.linspace(-110, -80, 8)},
        attrs={"history": "Created for test"},
    )
    options = {"coastlines": False, "extent": (-113, -78, 28, 43), "levels": 5}
    eager = SpatialContourPlot(data.copy(), coastlines=False)
    eager.plot(**options)

    runs = []
    try:
        with Callback(start=lambda graph: runs.append(len(graph))):
            lazy = SpatialContourPlot(
                data.chunk({"lat": 3, "lon": 4}), coastlines=False
            )
            assert not runs
            lazy.plot(**options)
        assert len(runs) == 1
        assert lazy.modelvar.chunks is not None
        assert "Created for test" in lazy.modelvar.attrs["history"]
        eager_contours, lazy_contours = (
            eager.ax.collections[-1],
            lazy.ax.collections[-1],
        )
        np.testing.assert_allclose(eager_contours.levels, lazy_contours.levels)
        for expected, actual in zip(
            eager_contours.get_paths(), lazy_contours.get_paths(), strict=True
        ):
            np.testing.assert_allclose(expected.vertices, actual.vertices)
        np.testing.assert_allclose(
            eager.fig.axes[-1].get_yticks(), lazy.fig.axes[-1].get_yticks()
        )
        np.testing.assert_allclose(lazy.ax.get_extent(), eager.ax.get_extent())
    finally:
        plt.close("all")
