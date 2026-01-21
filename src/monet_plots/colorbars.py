"""Colorbar helper functions"""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


def get_linear_scale(
    data, cmap="viridis", vmin=None, vmax=None, p_min=None, p_max=None
):
    """
    Get a linear colormap and normalization object.

    Parameters
    ----------
    data : array-like
        The data to scale.
    cmap : str or matplotlib.colors.Colormap, optional
        The colormap to use, by default "viridis".
    vmin : float, optional
        Minimum value for the scale. If None, uses min(data) or p_min.
    vmax : float, optional
        Maximum value for the scale. If None, uses max(data) or p_max.
    p_min : float, optional
        Percentile for minimum value (0-100).
    p_max : float, optional
        Percentile for maximum value (0-100).

    Returns
    -------
    tuple
        (colormap, Normalize)
    """
    if p_min is not None:
        vmin = np.nanpercentile(data, p_min)
    if p_max is not None:
        vmax = np.nanpercentile(data, p_max)

    if vmin is None:
        vmin = np.nanmin(data)
    if vmax is None:
        vmax = np.nanmax(data)

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    return cmap, norm


def get_diverging_scale(data, cmap="RdBu_r", center=0, span=None, p_span=None):
    """
    Get a diverging colormap and normalization object centered at a value.

    Parameters
    ----------
    data : array-like
        The data to scale.
    cmap : str or matplotlib.colors.Colormap, optional
        The colormap to use, by default "RdBu_r".
    center : float, optional
        The value to center the scale at, by default 0.
    span : float, optional
        The absolute range from the center (center +/- span).
    p_span : float, optional
        The percentile of absolute differences from center to use as span.

    Returns
    -------
    tuple
        (colormap, Normalize)
    """
    if span is not None:
        pass
    elif p_span is not None:
        diff = np.abs(data - center)
        span = np.nanpercentile(diff, p_span)
    else:
        span = np.nanmax(np.abs(data - center))

    vmin = center - span
    vmax = center + span

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    return cmap, norm


def get_discrete_scale(
    data, cmap="viridis", n_levels=10, vmin=None, vmax=None, extend="both"
):
    """
    Get a discrete colormap and BoundaryNorm with 'nice' numbers.

    Parameters
    ----------
    data : array-like
        The data to scale.
    cmap : str or matplotlib.colors.Colormap, optional
        The colormap to use, by default "viridis".
    n_levels : int, optional
        Target number of discrete levels, by default 10.
    vmin : float, optional
        Minimum value for the scale.
    vmax : float, optional
        Maximum value for the scale.
    extend : str, optional
        Whether to extend the scale ('neither', 'both', 'min', 'max'),
        by default "both".

    Returns
    -------
    tuple
        (colormap, BoundaryNorm)
    """
    if vmin is None:
        vmin = np.nanmin(data)
    if vmax is None:
        vmax = np.nanmax(data)

    locator = MaxNLocator(nbins=n_levels, steps=[1, 2, 2.5, 5, 10])
    levels = locator.tick_values(vmin, vmax)

    if isinstance(cmap, str):
        cmap_obj = plt.get_cmap(cmap)
    else:
        cmap_obj = cmap

    n_colors = len(levels) - 1
    discrete_cmap = cmap_discretize(cmap_obj, n_colors)

    norm = mcolors.BoundaryNorm(levels, ncolors=discrete_cmap.N, extend=extend)

    return discrete_cmap, norm


def get_log_scale(data, cmap="viridis", vmin=None, vmax=None):
    """
    Get a logarithmic colormap and normalization object.

    Parameters
    ----------
    data : array-like
        The data to scale.
    cmap : str or matplotlib.colors.Colormap, optional
        The colormap to use, by default "viridis".
    vmin : float, optional
        Minimum value for the scale (>0).
    vmax : float, optional
        Maximum value for the scale.

    Returns
    -------
    tuple
        (colormap, LogNorm)
    """
    data_positive = data[data > 0]
    if vmin is None:
        vmin = np.nanmin(data_positive) if data_positive.size > 0 else 1e-1
    if vmax is None:
        vmax = np.nanmax(data)

    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    return cmap, norm


def colorbar_index(
    ncolors,
    cmap,
    minval=None,
    maxval=None,
    dtype="int",
    basemap=None,
    ax=None,
    **kwargs,
):
    """Create a colorbar with discrete colors and custom tick labels.

    Parameters
    ----------
    ncolors : int
        Number of discrete colors to use in the colorbar.
    cmap : str or matplotlib.colors.Colormap
        Colormap to discretize and use for the colorbar.
    minval : float, optional
        Minimum value for the colorbar tick labels. If None and maxval is None,
        tick labels will range from 0 to ncolors. If None and maxval is provided,
        tick labels will range from 0 to maxval.
    maxval : float, optional
        Maximum value for the colorbar tick labels. If None, tick labels
        will range from 0 or minval to ncolors.
    dtype : str or type, default "int"
        Data type for tick label values (e.g., "int", "float").
    basemap : matplotlib.mpl_toolkits.basemap.Basemap, optional
        Basemap instance to attach the colorbar to. If None, uses plt.colorbar.
    ax : matplotlib.axes.Axes, optional
        Axes to attach the colorbar to. If None, uses plt.gca().
    **kwargs : Any
        Additional keyword arguments for plt.colorbar.

    Returns
    -------
    tuple
        (colorbar, discretized_cmap) where:
        - colorbar is the matplotlib.colorbar.Colorbar instance
        - discretized_cmap is the discretized colormap
    """
    import matplotlib.cm as cm

    cmap = cmap_discretize(cmap, ncolors)
    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array([])
    mappable.set_clim(-0.5, ncolors + 0.5)

    if basemap is not None:
        colorbar = basemap.colorbar(mappable, format="%1.2g")
    else:
        # Remove shrink=0.8 to allow for flexible sizing, matching axes by default
        # if used with appropriate arguments or through automated scaling.
        # We use a default fraction and pad that works well for many plots.
        cbar_kwargs = {"format": "%1.2g", "ax": ax, "fraction": 0.046, "pad": 0.04}
        cbar_kwargs.update(kwargs)
        colorbar = plt.colorbar(mappable, **cbar_kwargs)

    colorbar.set_ticks(np.linspace(0, ncolors, ncolors))
    if (minval is None) & (maxval is not None):
        colorbar.set_ticklabels(
            np.around(np.linspace(0, maxval, ncolors).astype(dtype), 2)
        )
    elif (minval is None) & (maxval is None):
        colorbar.set_ticklabels(
            np.around(np.linspace(0, ncolors, ncolors).astype(dtype), 2)
        )
    else:
        colorbar.set_ticklabels(
            np.around(np.linspace(minval, maxval, ncolors).astype(dtype), 2)
        )

    return colorbar, cmap


def cmap_discretize(cmap, N):
    """Return a discrete colormap from a continuous colormap.

    Creates a new colormap by discretizing an existing continuous colormap
    into N distinct colors while preserving the color transitions.

    Parameters
    ----------
    cmap : str or matplotlib.colors.Colormap
        Colormap instance or registered colormap name to discretize.
        Example: cm.jet, 'viridis', etc.
    N : int
        Number of discrete colors to use in the new colormap.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        A new colormap object with N discrete colors based on the input colormap.
        The name will be the original colormap name with "_N" appended.
    """
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0, 1.0, N), (0.0, 0.0, 0.0, 0.0)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1.0, N + 1)
    cdict = {}
    for ki, key in enumerate(("red", "green", "blue")):
        cdict[key] = [
            (indices[i], colors_rgba[i - 1, ki], colors_rgba[i, ki])
            for i in range(N + 1)
        ]
    # Return colormap object.
    return mcolors.LinearSegmentedColormap(cmap.name + "_%d" % N, cdict, 1024)
