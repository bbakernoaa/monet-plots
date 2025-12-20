import pytest
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
import matplotlib.pyplot as plt
from monet_plots.plots.spatial import SpatialPlot


@pytest.fixture
def clear_figures():
    """Clear all existing figures before and after a test."""
    plt.close("all")
    yield
    plt.close("all")


def test_spatial_plot_returns_axes(clear_figures):
    """Test that SpatialPlot creates a GeoAxes instance."""
    plot = SpatialPlot()
    assert isinstance(plot.ax, GeoAxes)


def test_spatial_plot_returns_fig_and_axes(clear_figures):
    """Test that SpatialPlot creates a figure and axes."""
    plot = SpatialPlot()
    assert isinstance(plot.fig, plt.Figure)
    assert isinstance(plot.ax, GeoAxes)


def test_spatial_plot_projection(clear_figures):
    """Test that SpatialPlot sets the projection correctly."""
    projection = ccrs.Mollweide()
    plot = SpatialPlot(projection=projection)
    assert isinstance(plot.ax.projection, ccrs.Mollweide)


def test_spatial_plot_extent(clear_figures):
    """Test that SpatialPlot sets the extent correctly."""
    extent = [-120, -60, 20, 50]
    plot = SpatialPlot(extent=extent)
    plot.plot()  # Extent is set in _draw_features, which is called by plot
    assert plot.ax.get_extent() == pytest.approx(tuple(extent), abs=4)


def test_draw_map_with_features(clear_figures):
    """Test that SpatialPlot adds features like coastlines, states, and countries."""
    plot = SpatialPlot(coastlines=True, states=True, countries=True, resolution="110m")
    plot.plot()
    assert len(plot.ax.collections) > 0
