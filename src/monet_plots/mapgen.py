"""Map utilities."""

import cartopy.crs as ccrs
import cartopy.feature as cfeature


def _setup_map_projection(crs, kwargs):
    """Set up map projection in kwargs."""
    if "subplot_kw" not in kwargs:
        kwargs["subplot_kw"] = {}

    if crs is not None:
        kwargs["subplot_kw"]["projection"] = crs
    elif "projection" not in kwargs["subplot_kw"]:
        kwargs["subplot_kw"]["projection"] = ccrs.PlateCarree()


def _add_natural_earth_features(ax):
    """Add natural earth features to the map."""
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.LAKES)
    ax.add_feature(cfeature.RIVERS)


def _create_boundary_features(resolution):
    """Create state and county boundary features."""
    states_provinces = cfeature.NaturalEarthFeature(
        category="cultural",
        name="admin_1_states_provinces_lines",
        scale=resolution,
        facecolor="none",
        edgecolor="k",
    )

    counties = cfeature.NaturalEarthFeature(
        category="cultural",
        name="admin_2_counties",
        scale=resolution,
        facecolor="none",
        edgecolor="k",
    )

    return states_provinces, counties


def _add_boundary_features(ax, coastlines, states, counties, countries, resolution, linewidth):
    """Add boundary features to the map."""
    if coastlines:
        ax.coastlines(resolution, linewidth=linewidth)

    if countries:
        ax.add_feature(cfeature.BORDERS, linewidth=linewidth)

    if states or counties:
        states_provinces, county_features = _create_boundary_features(resolution)

        if states:
            ax.add_feature(states_provinces, linewidth=linewidth)

        if counties:
            ax.add_feature(county_features, linewidth=linewidth)


def draw_map(
    *,
    crs=None,
    natural_earth=False,
    coastlines=True,
    states=False,
    counties=False,
    countries=True,
    resolution="10m",
    extent=None,
    figsize=(10, 5),
    linewidth=0.25,
    return_fig=False,
    **kwargs,
):
    """Draw a map with Cartopy.

    .. deprecated::
        This function is deprecated. Use `SpatialPlot.draw_map()` instead.

    Creates a map using Cartopy with configurable features like coastlines,
    borders, and natural earth elements.
    """
    import warnings

    warnings.warn("draw_map is deprecated. Use SpatialPlot.draw_map() instead.", DeprecationWarning, stacklevel=2)

    from .plots.spatial import SpatialPlot

    return SpatialPlot.draw_map(
        crs=crs,
        natural_earth=natural_earth,
        coastlines=coastlines,
        states=states,
        counties=counties,
        countries=countries,
        resolution=resolution,
        extent=extent,
        figsize=figsize,
        linewidth=linewidth,
        return_fig=return_fig,
        **kwargs,
    )
