import pytest
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
import matplotlib.pyplot as plt
from monet_plots.mapgen import draw_map


@pytest.fixture
def clear_figures():
    """Clear all existing figures before and after a test."""
    plt.close("all")
    yield
    plt.close("all")


def test_draw_map_returns_axes(clear_figures):
    """Test that draw_map returns a GeoAxes instance."""
    ax = draw_map()
    assert isinstance(ax, GeoAxes)


def test_draw_map_returns_fig_and_axes(clear_figures):
    """Test that draw_map returns a figure and axes when return_fig is True."""
    fig, ax = draw_map(return_fig=True)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, GeoAxes)


def test_draw_map_projection(clear_figures):
    """Test that draw_map sets the projection correctly."""
    projection = ccrs.Mollweide()
    ax = draw_map(crs=projection)
    assert isinstance(ax.projection, ccrs.Mollweide)


def test_draw_map_extent(clear_figures):
    """Test that draw_map sets the extent correctly."""
    extent = [-120, -60, 20, 50]
    ax = draw_map(extent=extent)
    assert ax.get_extent() == pytest.approx(tuple(extent), abs=4)


@pytest.mark.mpl_image_compare
def test_draw_map_with_features(clear_figures):
    """Test that draw_map adds features like coastlines, states, and countries."""
    fig, ax = draw_map(coastlines=True, states=True, countries=True, resolution="110m", return_fig=True)
    return fig
