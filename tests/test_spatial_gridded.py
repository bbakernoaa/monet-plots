import numpy as np
import pytest
import xarray as xr
import matplotlib.pyplot as plt
from monet_plots.plots.spatial_imshow import SpatialImshowPlot
from monet_plots.plots.spatial_contour import SpatialContourPlot
from monet_plots.plots.facet_grid import SpatialFacetGridPlot


def test_imshow_robust_coords():
    """Test SpatialImshowPlot with various coordinate names and attributes."""
    # Test case 1: 'latitude' and 'longitude' names
    lat = np.linspace(30, 40, 10)
    lon = np.linspace(-100, -90, 10)
    da = xr.DataArray(
        np.random.rand(10, 10),
        coords={"latitude": lat, "longitude": lon},
        dims=["latitude", "longitude"],
    )
    p = SpatialImshowPlot(da)
    ax = p.plot()
    extent = ax.images[0].get_extent()
    assert extent[0] == -100.0
    assert extent[1] == -90.0
    plt.close(p.fig)

    # Test case 2: 'y' and 'x' names with units
    y = np.linspace(30, 40, 10)
    x = np.linspace(-100, -90, 10)
    da2 = xr.DataArray(np.random.rand(10, 10), coords={"y": y, "x": x}, dims=["y", "x"])
    da2.y.attrs["units"] = "degrees_north"
    da2.x.attrs["units"] = "degrees_east"
    p2 = SpatialImshowPlot(da2)
    ax2 = p2.plot()
    extent2 = ax2.images[0].get_extent()
    assert extent2[0] == -100.0
    assert extent2[2] == 30.0
    plt.close(p2.fig)


def test_imshow_monotonic_enforcement():
    """Verify that latitude is forced to be monotonic increasing."""
    lat = np.linspace(50, 30, 10)  # Decreasing
    lon = np.linspace(-100, -90, 10)
    data = np.random.rand(10, 10)
    da = xr.DataArray(data, coords={"lat": lat, "lon": lon}, dims=["lat", "lon"])

    p = SpatialImshowPlot(da)
    ax = p.plot()
    img = ax.images[0]
    extent = img.get_extent()

    # Extent should be [min_lon, max_lon, min_lat, max_lat]
    assert extent[2] == 30.0
    assert extent[3] == 50.0

    # Data should have been flipped
    model_data = img.get_array()
    assert np.allclose(model_data[0, :], data[-1, :])
    plt.close(p.fig)


def test_contour_xarray():
    """Test SpatialContourPlot with xr.DataArray."""
    lat = np.linspace(30, 40, 10)
    lon = np.linspace(-100, -90, 10)
    da = xr.DataArray(
        np.random.rand(10, 10), coords={"lat": lat, "lon": lon}, dims=["lat", "lon"]
    )

    p = SpatialContourPlot(da, cmap="viridis", levels=10)
    ax = p.plot()
    assert len(ax.collections) > 0  # Should have contour collections
    plt.close(p.fig)


def test_facet_grid_xarray_auto_coords():
    """Test SpatialFacetGridPlot automatically finds coordinates in DataArray."""
    time = [0, 1]
    latitude = np.linspace(30, 40, 10)
    longitude = np.linspace(-100, -90, 10)
    da = xr.DataArray(
        np.random.rand(2, 10, 10),
        coords={"time": time, "latitude": latitude, "longitude": longitude},
        dims=["time", "latitude", "longitude"],
        name="val",
    )

    sfg = SpatialFacetGridPlot(da, col="time")
    # Should use 'latitude' and 'longitude' automatically
    sfg.map_monet(SpatialImshowPlot)

    for ax in sfg.axs_flattened:
        if ax is not None:
            assert len(ax.images) > 0
            extent = ax.images[0].get_extent()
            assert extent[0] == -100.0
            assert extent[2] == 30.0
    plt.close(sfg.fig)


def test_imshow_dataarray_faceting_redirect():
    """Verify that SpatialImshowPlot(DataArray, col='...') redirects to SpatialFacetGridPlot."""
    time = [0, 1]
    latitude = np.linspace(30, 40, 10)
    longitude = np.linspace(-100, -90, 10)
    da = xr.DataArray(
        np.random.rand(2, 10, 10),
        coords={"time": time, "latitude": latitude, "longitude": longitude},
        dims=["time", "latitude", "longitude"],
        name="val",
    )

    # Calling SpatialImshowPlot with 'col' should return a SpatialFacetGridPlot result
    # which is the result of map_monet
    res = SpatialImshowPlot(da, col="time")

    # In my implementation, SpatialFacetGridPlot.map_monet returns self
    from monet_plots.plots.facet_grid import SpatialFacetGridPlot

    assert isinstance(res, SpatialFacetGridPlot)
    assert len(res.axs_flattened) == 2
    plt.close(res.fig)


def test_constructor_kwargs_persistence():
    """Verify that kwargs passed to constructor are used in plot()."""
    da = xr.DataArray(
        np.random.rand(10, 10),
        coords={"lat": np.arange(10), "lon": np.arange(10)},
        dims=["lat", "lon"],
    )

    # cmap in constructor
    p = SpatialImshowPlot(da, cmap="turbo")
    ax = p.plot()
    assert ax.images[0].get_cmap().name == "turbo"
    plt.close(p.fig)

    # vmin/vmax in constructor
    p2 = SpatialImshowPlot(da, vmin=0.1, vmax=0.9)
    ax2 = p2.plot()
    assert ax2.images[0].get_clim() == (0.1, 0.9)
    plt.close(p2.fig)


if __name__ == "__main__":
    pytest.main([__file__])
