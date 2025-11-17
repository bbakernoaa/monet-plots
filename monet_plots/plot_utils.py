import warnings


def _dynamic_fig_size(obj):
    """Try to determine a generic figure size based on the shape of obj

    Parameters
    ----------
    obj : A 2D xarray DataArray
        Description of parameter `obj`.

    Returns
    -------
    type
        Description of returned object.

    """
    if "x" in obj.dims:
        nx, ny = len(obj.x), len(obj.y)
        scale = float(ny) / float(nx)
    elif "latitude" in obj.dims:
        nx, ny = len(obj.longitude), len(obj.latitude)
        scale = float(ny) / float(nx)
    elif "lat" in obj.dims:
        nx, ny = len(obj.lon), len(obj.lat)
        scale = float(ny) / float(nx)
    figsize = (10, 10 * scale)
    return figsize


def _set_outline_patch_alpha(ax, alpha=0):
    """Set the transparency of map outline patches for Cartopy GeoAxes.

    This function attempts multiple methods to set the alpha (transparency) of
    map outlines when using Cartopy, handling different versions and configurations.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or cartopy.mpl.geoaxes.GeoAxes
        The axes object whose outline transparency should be modified.
    alpha : float, default 0
        Alpha value between 0 (fully transparent) and 1 (fully opaque).

    Notes
    -----
    The function tries multiple approaches to accommodate different Cartopy versions
    and configurations. If all attempts fail, a warning is issued.
    """
    for f in [
        lambda alpha: ax.axes.outline_patch.set_alpha(alpha),
        lambda alpha: ax.outline_patch.set_alpha(alpha),
        lambda alpha: ax.spines["geo"].set_alpha(alpha),
    ]:
        try:
            f(alpha)
        except AttributeError:
            continue
        else:
            break
    else:
        warnings.warn("unable to set outline_patch alpha", stacklevel=2)