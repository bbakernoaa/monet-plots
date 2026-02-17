import pytest
import xarray as xr
import numpy as np
from monet_plots.plots.spatial_imshow import SpatialImshowPlot
from monet_plots.plots.spatial_contour import SpatialContourPlot
from monet_plots.plots.facet_grid import SpatialFacetGridPlot


@pytest.fixture
def sample_da():
    """Create a sample 3D xarray DataArray for faceting tests."""
    lon = np.linspace(-120, -70, 10)
    lat = np.linspace(20, 50, 10)
    time = [0, 1]
    data = np.random.rand(2, 10, 10)
    da = xr.DataArray(
        data,
        coords={"time": time, "lat": lat, "lon": lon},
        dims=("time", "lat", "lon"),
        name="test_var",
    )
    da.lat.attrs["units"] = "degrees_north"
    da.lon.attrs["units"] = "degrees_east"
    return da


def test_spatial_imshow_faceting(sample_da):
    """Test SpatialImshowPlot redirection to SpatialFacetGridPlot."""
    plot = SpatialImshowPlot(sample_da, col="time")
    assert isinstance(plot, SpatialFacetGridPlot)
    assert plot.col == "time"

    # Test plotting
    grid = plot.plot()
    assert grid is not None
    assert hasattr(grid, "axs")


def test_spatial_contour_faceting(sample_da):
    """Test SpatialContourPlot redirection to SpatialFacetGridPlot."""
    plot = SpatialContourPlot(sample_da, col="time")
    assert isinstance(plot, SpatialFacetGridPlot)

    # Test plotting
    grid = plot.plot()
    assert grid is not None


def test_spatial_facet_grid_features(sample_da):
    """Test adding features to all facets."""
    plot = SpatialFacetGridPlot(sample_da, col="time")
    plot.plot(coastlines=True, states=True)
    # No easy way to assert features were added without deep inspection
    # but at least it shouldn't crash.


def test_spatial_imshow_xarray_single(sample_da):
    """Test SpatialImshowPlot with xarray DataArray but single panel."""
    da_single = sample_da.isel(time=0)
    plot = SpatialImshowPlot(da_single)
    assert not isinstance(plot, SpatialFacetGridPlot)
    ax = plot.plot()
    assert ax is not None


def test_spatial_contour_xarray_single(sample_da):
    """Test SpatialContourPlot with xarray DataArray but single panel."""
    da_single = sample_da.isel(time=0)
    plot = SpatialContourPlot(da_single)
    assert not isinstance(plot, SpatialFacetGridPlot)
    ax = plot.plot()
    assert ax is not None


def test_xarray_accessor_imshow(sample_da):
    """Test the mplots xarray accessor for imshow."""
    grid = sample_da.mplots.imshow(col="time")
    assert isinstance(grid, xr.plot.FacetGrid)


def test_xarray_accessor_contourf(sample_da):
    """Test the mplots xarray accessor for contourf."""
    grid = sample_da.mplots.contourf(col="time")
    assert isinstance(grid, xr.plot.FacetGrid)


def test_xarray_accessor_contour(sample_da):
    """Test the mplots xarray accessor for contour."""
    grid = sample_da.mplots.contour(col="time")
    assert isinstance(grid, xr.plot.FacetGrid)


def test_xarray_accessor_pcolormesh(sample_da):
    """Test the mplots xarray accessor for pcolormesh."""
    grid = sample_da.mplots.pcolormesh(col="time")
    assert isinstance(grid, xr.plot.FacetGrid)


def test_xarray_accessor_scatter(sample_da):
    """Test the mplots xarray accessor for scatter."""
    # Reshape data for scatter if needed or just use as is
    grid = sample_da.mplots.scatter(col="time")
    assert isinstance(grid, xr.plot.FacetGrid)


def test_xarray_accessor_track(sample_da):
    """Test the mplots xarray accessor for track."""
    # SpatialTrack expects specifically a trajectory DataArray
    lon = np.linspace(-120, -70, 10)
    lat = np.linspace(20, 50, 10)
    data = np.random.rand(10)
    da_track = xr.DataArray(
        data,
        coords={"lat": ("points", lat), "lon": ("points", lon)},
        dims="points",
        name="track_var",
    )
    artist = da_track.mplots.track()
    assert artist is not None


def test_xarray_accessor_single(sample_da):
    """Test the mplots xarray accessor for single panel."""
    da_single = sample_da.isel(time=0)
    ax = da_single.mplots.imshow()
    assert ax is not None
