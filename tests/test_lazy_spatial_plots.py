import numpy as np
import xarray as xr
from monet_plots.plots.spatial_imshow import SpatialImshowPlot
from monet_plots.plots.spatial_contour import SpatialContourPlot
from monet_plots.plots.spatial import SpatialTrack


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
    assert ax_lazy is not None

    plt.close("all")
