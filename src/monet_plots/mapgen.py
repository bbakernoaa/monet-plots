"""Map utilities."""

import warnings


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
        This function is deprecated. Use `SpatialPlot()` instead.

    Creates a map using Cartopy with configurable features like coastlines,
    borders, and natural earth elements.
    """
    warnings.warn(
        "draw_map is deprecated. Use SpatialPlot() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

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
