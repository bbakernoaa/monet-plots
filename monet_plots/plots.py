"""plotting routines"""

import functools

import matplotlib.pyplot as plt
import seaborn as sns

from . import taylordiagram as td
from .colorbars import colorbar_index
from .plot_utils import _set_outline_patch_alpha

# colors = ['#1e90ff','#045C5C','#00A847','#DB4291','#BB7E5D']
colors = ["#1e90ff", "#DA70D6", "#228B22", "#FA8072", "#FF1493"]


def _default_sns_context(f):
    """Decorator to apply a default seaborn context and color palette to plotting functions."""

    @functools.wraps(f)
    def inner(*args, **kwargs):
        with sns.plotting_context("poster"), sns.color_palette(colors):
            return f(*args, **kwargs)

    return inner


# Spatial Plots
@_default_sns_context
def make_spatial_plot(modelvar, m, dpi=None, plotargs={}, ncolors=15, discrete=False):
    """Create a basic spatial plot using imshow.

    Parameters
    ----------
    modelvar : numpy.ndarray
        2D model variable array to plot.
    m : mpl_toolkits.basemap.Basemap
        Basemap instance for mapping.
    dpi : int, optional
        Dots per inch for the figure. Higher values increase resolution.
    plotargs : dict, default {}
        Additional arguments to pass to imshow. Common options include 'cmap',
        'vmin', 'vmax', and 'alpha'.
    ncolors : int, default 15
        Number of discrete colors when using discrete colorbar.
    discrete : bool, default False
        If True, use a discrete colorbar instead of a continuous one.

    Returns
    -------
    tuple
        (figure, axes, colorbar, colormap, vmin, vmax)
        - figure: matplotlib Figure instance
        - axes: matplotlib Axes instance
        - colorbar: matplotlib Colorbar instance
        - colormap: matplotlib Colormap instance
        - vmin, vmax: minimum and maximum values for the colormap
    """
    f, ax = plt.subplots(1, 1, figsize=(11, 6), frameon=False)
    # determine colorbar
    if "cmap" not in plotargs:
        plotargs["cmap"] = "viridis"
    if discrete and "vmin" in plotargs and "vmax" in plotargs:
        c, cmap = colorbar_index(
            ncolors, plotargs["cmap"], minval=plotargs["vmin"], maxval=plotargs["vmax"], basemap=m
        )
        plotargs["cmap"] = cmap
        m.imshow(modelvar, **plotargs)
        vmin, vmax = plotargs["vmin"], plotargs["vmax"]
    elif discrete:
        temp = m.imshow(modelvar, **plotargs)
        vmin, vmax = temp.get_clim()
        c, cmap = colorbar_index(ncolors, plotargs["cmap"], minval=vmin, maxval=vmax, basemap=m)
        plotargs["cmap"] = cmap
        m.imshow(modelvar, vmin=vmin, vmax=vmax, **plotargs)
    else:
        temp = m.imshow(modelvar, **plotargs)
        c = m.colorbar()
        vmin, vmax = temp.get_clim()
        cmap = plotargs["cmap"]
    # draw borders
    m.drawstates()
    m.drawcoastlines(linewidth=0.3)
    m.drawcountries()
    return f, ax, c, cmap, vmin, vmax


@_default_sns_context
def spatial(modelvar, **kwargs):
    """Create a simple spatial plot from an xarray object.

    A convenience wrapper for xarray's plot method with consistent styling.

    Parameters
    ----------
    modelvar : xarray.DataArray
        The data to plot spatially.
    **kwargs
        Additional keyword arguments passed to xarray's plot method.
        If 'ax' is not provided, a new figure and axes will be created.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.
    """
    if kwargs.get("ax") is None:
        f, ax = plt.subplots(1, 1, figsize=(11, 6), frameon=False)
        kwargs["ax"] = ax
    ax = modelvar.plot(**kwargs)
    plt.tight_layout()
    return ax


@_default_sns_context
def make_spatial_contours(
    modelvar,
    gridobj,
    date,
    m,
    dpi=None,
    savename="",
    discrete=True,
    ncolors=None,
    dtype="int",
    **kwargs,
):
    """Create a contour plot on a map with optional discrete colorbar.

    Parameters
    ----------
    modelvar : numpy.ndarray
        2D model variable array to contour.
    gridobj : object
        Object containing grid information with LAT and LON variables.
    date : datetime.datetime
        Date/time for the plot title.
    m : mpl_toolkits.basemap.Basemap
        Basemap instance for mapping.
    dpi : int, optional
        Dots per inch for the figure if saving.
    savename : str, default ""
        If provided, save the figure to this path with date appended.
    discrete : bool, default True
        If True, use a discrete colorbar instead of a continuous one.
    ncolors : int, optional
        Number of discrete colors when using discrete colorbar.
    dtype : str, default "int"
        Data type for colorbar tick labels.
    **kwargs
        Additional arguments to pass to contourf. Must include 'cmap' and 'levels'.

    Returns
    -------
    matplotlib.colorbar.Colorbar
        The colorbar instance.
    """
    plt.figure(figsize=(11, 6), frameon=False)
    lat = gridobj.variables["LAT"][0, 0, :, :].squeeze()
    lon = gridobj.variables["LON"][0, 0, :, :].squeeze()
    # define map and draw boundaries
    m.drawstates()
    m.drawcoastlines(linewidth=0.3)
    m.drawcountries()
    x, y = m(lon, lat)
    plt.axis("off")
    m.contourf(x, y, modelvar, **kwargs)
    cmap = kwargs["cmap"]
    levels = kwargs["levels"]
    if discrete:
        c, cmap = colorbar_index(
            ncolors, cmap, minval=levels[0], maxval=levels[-1], basemap=m, dtype=dtype
        )
    else:
        c = m.colorbar()
    titstring = date.strftime("%B %d %Y %H")
    plt.title(titstring)

    plt.tight_layout()
    if savename != "":
        plt.savefig(savename + date.strftime("%Y%m%d_%H.jpg"), dpi=dpi)
        plt.close()
    return c


@_default_sns_context
def wind_quiver(ws, wdir, gridobj, m, **kwargs):
    """Create a quiver plot of wind vectors on a map.

    Parameters
    ----------
    ws : numpy.ndarray
        2D array of wind speeds.
    wdir : numpy.ndarray
        2D array of wind directions (meteorological convention, degrees).
    gridobj : object
        Object containing grid information with LAT and LON variables.
    m : mpl_toolkits.basemap.Basemap
        Basemap instance for mapping.
    **kwargs
        Additional arguments to pass to quiver. Common options include
        'scale', 'scale_units', and 'width'.

    Returns
    -------
    matplotlib.quiver.Quiver
        The quiver instance.
    """
    from . import tools

    lat = gridobj.variables["LAT"][0, 0, :, :].squeeze()
    lon = gridobj.variables["LON"][0, 0, :, :].squeeze()
    # define map and draw boundaries
    x, y = m(lon, lat)
    u, v = tools.wsdir2uv(ws, wdir)
    quiv = m.quiver(x[::15, ::15], y[::15, ::15], u[::15, ::15], v[::15, ::15], **kwargs)
    return quiv


@_default_sns_context
def wind_barbs(ws, wdir, gridobj, m, **kwargs):
    """Create a barbs plot of wind on a map.

    Parameters
    ----------
    ws : numpy.ndarray
        2D array of wind speeds.
    wdir : numpy.ndarray
        2D array of wind directions (meteorological convention, degrees).
    gridobj : object
        Object containing grid information with LAT and LON variables.
    m : mpl_toolkits.basemap.Basemap
        Basemap instance for mapping.
    **kwargs
        Additional arguments to pass to barbs. Common options include
        'length', 'pivot', and 'barb_increments'.

    Returns
    -------
    None
    """
    import tools

    lat = gridobj.variables["LAT"][0, 0, :, :].squeeze()
    lon = gridobj.variables["LON"][0, 0, :, :].squeeze()
    # define map and draw boundaries
    x, y = m(lon, lat)
    u, v = tools.wsdir2uv(ws, wdir)
    m.barbs(x[::15, ::15], y[::15, ::15], u[::15, ::15], v[::15, ::15], **kwargs)


def normval(vmin, vmax, cmap):
    """Create a BoundaryNorm for discrete colormaps with specific bounds.

    Parameters
    ----------
    vmin : float
        Minimum value for the colormap.
    vmax : float
        Maximum value for the colormap.
    cmap : matplotlib.colors.Colormap
        The colormap to create bounds for.

    Returns
    -------
    matplotlib.colors.BoundaryNorm
        A boundary norm with evenly spaced bounds from vmin to vmax in steps of 5.0.
    """
    from matplotlib.colors import BoundaryNorm
    from numpy import arange

    bounds = arange(vmin, vmax + 5.0, 5.0)
    norm = BoundaryNorm(boundaries=bounds, ncolors=cmap.N)
    return norm


@_default_sns_context
def spatial_bias_scatter(
    df, m, date, vmin=None, vmax=None, savename="", ncolors=15, fact=1.5, cmap="RdBu_r"
):
    """Create a scatter plot showing bias between model and observations on a map.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing 'latitude', 'longitude', 'CMAQ', and 'Obs' columns.
    m : mpl_toolkits.basemap.Basemap
        Basemap instance for mapping.
    date : str or datetime.datetime
        Date to filter the DataFrame. Only entries matching this date will be plotted.
    vmin : float, optional
        Minimum value for colorscale. If None, automatically determined.
    vmax : float, optional
        Maximum value for colorscale. If None, automatically determined.
    savename : str, default ""
        If provided, save the figure to this path with date appended.
    ncolors : int, default 15
        Number of discrete colors for the colorbar.
    fact : float, default 1.5
        Scaling factor for point sizes.
    cmap : str or matplotlib.colors.Colormap, default "RdBu_r"
        Colormap to use for bias values.

    Returns
    -------
    tuple
        (figure, axes, colorbar) containing the matplotlib objects.

    Notes
    -----
    The scatter points are colored by the difference (CMAQ - Obs) and sized
    by the absolute magnitude of this difference, making larger biases more visible.
    """
    from numpy import around
    from scipy.stats import scoreatpercentile as score

    #    plt.figure(figsize=(11, 6), frameon=False)
    f, ax = plt.subplots(figsize=(11, 6), frameon=False)
    ax.set_facecolor("white")
    diff = df.CMAQ - df.Obs
    top = around(score(diff.abs(), per=95))
    new = df[df.datetime == date]
    x, y = m(new.longitude.values, new.latitude.values)
    c, cmap = colorbar_index(ncolors, cmap, minval=top * -1, maxval=top, basemap=m)

    c.ax.tick_params(labelsize=13)
    #    cmap = cmap_discretize(cmap, ncolors)
    colors = new.CMAQ - new.Obs
    ss = (new.CMAQ - new.Obs).abs() / top * 100.0
    ss[ss > 300] = 300.0
    plt.scatter(
        x,
        y,
        c=colors,
        s=ss,
        vmin=-1.0 * top,
        vmax=top,
        cmap=cmap,
        edgecolors="k",
        linewidths=0.25,
        alpha=0.7,
    )

    if savename != "":
        plt.savefig(savename + date + ".jpg", dpi=75.0)
        plt.close()
    return f, ax, c


@_default_sns_context
def timeseries(
    df,
    x="time",
    y="obs",
    ax=None,
    plotargs={},
    fillargs={"alpha": 0.2},
    title="",
    ylabel=None,
    label=None,
):
    """Create a timeseries plot with shaded error bounds.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data to plot.
    x : str, default "time"
        Column name to use for the x-axis (time).
    y : str, default "obs"
        Column name to use for the y-axis (values to plot).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure and axes.
    plotargs : dict, default {}
        Additional arguments to pass to DataFrame.plot().
    fillargs : dict, default {"alpha": 0.2}
        Additional arguments to pass to fill_between for the error shading.
    title : str, default ""
        Title for the plot.
    ylabel : str, optional
        Y-axis label. If None, uses variable name and units from DataFrame.
    label : str, optional
        Label for the plotted line (for legend). If None, uses y.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.

    Notes
    -----
    This function groups the data by time, plots the mean values, and adds
    shading for ±1 standard deviation around the mean.
    """
    with sns.axes_style("ticks"):
        if ax is None:
            f, ax = plt.subplots(figsize=(11, 6), frameon=False)
        df.index = df[x]
        m = df.groupby("time").mean()  # mean values for each sample time period
        e = df.groupby("time").std()  # std values for each sample time period
        variable = df.variable[0]
        if df.columns.isin(["units"]).max():
            unit = df.units[0]
        else:
            unit = "None"
        upper = m[y] + e[y]
        lower = m[y] - e[y]
        lower.loc[lower < 0] = 0
        lower = lower.values
        if "alpha" not in fillargs:
            fillargs["alpha"] = 0.2
        if label is not None:
            m.rename(columns={y: label}, inplace=True)
        else:
            label = y
        m[label].plot(ax=ax, **plotargs)
        ax.fill_between(m[label].index, lower, upper, **fillargs)
        if ylabel is None:
            ax.set_ylabel(variable + " (" + unit + ")")
        else:
            ax.set_ylabel(label)
        ax.set_xlabel("")
        plt.legend()
        plt.title(title)
        plt.tight_layout()

    return ax


@_default_sns_context
def kdeplot(df, title=None, label=None, ax=None, **kwargs):
    """Create a kernel density estimate plot.

    Parameters
    ----------
    df : pandas.Series or array-like
        Data to plot the distribution of.
    title : str, optional
        Title for the plot.
    label : str, optional
        Label for the plotted line (for legend).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure and axes.
    **kwargs
        Additional arguments passed to seaborn's kdeplot.
        Common options include 'shade', 'bw', and 'color'.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.
    """
    with sns.axes_style("ticks"):
        if ax is None:
            f, ax = plt.subplots(figsize=(11, 6), frameon=False)
            sns.despine()
        ax = sns.kdeplot(df, ax=ax, label=label, **kwargs)

    return ax


@_default_sns_context
def scatter(df, x=None, y=None, title=None, label=None, ax=None, **kwargs):
    """Create a scatter plot with regression line.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data to plot.
    x : str, optional
        Column name for x-axis values.
    y : str, optional
        Column name for y-axis values.
    title : str, optional
        Title for the plot.
    label : str, optional
        Label for the plot (for legend).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure and axes.
    **kwargs
        Additional arguments passed to seaborn's regplot.
        Common options include 'scatter_kws', 'line_kws', and 'ci'.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.
    """
    with sns.axes_style("ticks"):
        if ax is None:
            f, ax = plt.subplots(figsize=(8, 6), frameon=False)
        ax = sns.regplot(data=df, x=x, y=y, label=label, **kwargs)
        plt.title(title)

    return ax


@_default_sns_context
def create_taylor_diagram(
    df,
    marker="o",
    col1="obs",
    col2="model",
    label1="OBS",
    label2="MODEL",
    scale=1.5,
    addon=False,
    dia=None,
):
    """
    :no-index:

    Create a DataFrame-based Taylor diagram using the TaylorDiagram class.

    A convenience wrapper for easily creating Taylor diagrams from DataFrames.
    For the main Taylor diagram implementation, see :mod:`monet.plots.taylordiagram`.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing observation and model data
    marker : str, default "o"
        Marker style for plotting model points
    col1 : str, default "obs"
        Column name for observations
    col2 : str, default "model"
        Column name for model predictions
    label1 : str, default "OBS"
        Label for observations in legend
    label2 : str, default "MODEL"
        Label for model in legend
    scale : float, default 1.5
        Scale factor for diagram
    addon : bool, default False
        If True, add to existing diagram; if False, create new
    dia : TaylorDiagram, optional
        Existing diagram to add to if addon=True

    Returns
    -------
    TaylorDiagram
        The Taylor diagram instance
    """
    # Same implementation as before
    from numpy import corrcoef

    df = df.drop_duplicates().dropna(subset=[col1, col2])

    if not addon and dia is None:
        with sns.axes_style("ticks"):
            f = plt.figure(figsize=(12, 10))
            obsstd = df[col1].std()

            dia = td.TaylorDiagram(obsstd, scale=scale, fig=f, rect=111, label=label1)
            plt.grid(linewidth=1, alpha=0.5)
            cc = corrcoef(df[col1].values, df[col2].values)[0, 1]
            dia.add_sample(df[col2].std(), cc, marker=marker, zorder=9, ls=None, label=label2)
            contours = dia.add_contours(colors="0.5")
            plt.clabel(contours, inline=1, fontsize=10)
            plt.grid(alpha=0.5)
            plt.legend(fontsize="small", loc="best")

    elif not addon and dia is not None:
        print("Do you want to add this on? if so please turn the addon keyword to True")
    elif addon and dia is None:
        print("Please pass the previous Taylor Diagram Instance with dia keyword...")
    else:
        cc = corrcoef(df.Obs.values, df.CMAQ.values)[0, 1]
        dia.add_sample(df.CMAQ.std(), cc, marker=marker, zorder=9, ls=None, label=label1)
        plt.legend(fontsize="small", loc="best")
        plt.tight_layout()
    return dia


def sp_scatter_bias(
    df,
    col1=None,
    col2=None,
    ax=None,
    outline=False,
    tight=True,
    global_map=True,
    map_kwargs={},
    cbar_kwargs={},
    val_max=None,
    val_min=None,
    **kwargs,
):
    """Create a spatial scatter plot showing the bias (difference) between two columns in a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing latitude, longitude, and data columns to compare.
    col1 : str
        Name of the first column (reference value).
    col2 : str
        Name of the second column (comparison value).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new map using draw_map.
    outline : bool, default False
        Whether to show the map outline.
    tight : bool, default True
        Whether to apply tight_layout to the figure.
    global_map : bool, default True
        Whether to set global map boundaries (-180 to 180 longitude, -90 to 90 latitude).
    map_kwargs : dict, default {}
        Keyword arguments passed to draw_map if creating a new map.
    cbar_kwargs : dict, default {}
        Keyword arguments for colorbar customization.
    val_max : float, optional
        Maximum value for color scaling. If None, uses 95th percentile of absolute differences.
    val_min : float, optional
        Minimum value for color scaling (not currently used).
    **kwargs : dict
        Additional keyword arguments passed to DataFrame.plot.scatter.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object containing the plot.

    Notes
    -----
    The point size is scaled by the magnitude of the difference between col2 and col1,
    making larger differences more visually prominent. Differences are capped at 300 units
    for display purposes.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import scoreatpercentile as score

    if ax is None:
        ax = draw_map(**map_kwargs)
    try:
        if col1 is None or col2 is None:
            print("User must specify col1 and col2 in the dataframe")
            raise ValueError
        else:
            dfnew = df[["latitude", "longitude", col1, col2]].dropna().copy(deep=True)
            dfnew["sp_diff"] = dfnew[col2] - dfnew[col1]
            top = score(dfnew["sp_diff"].abs(), per=95)
            if val_max is not None:
                top = val_max
            # x, y = df.longitude.values, df.latitude.values
            dfnew["sp_diff_size"] = dfnew["sp_diff"].abs() / top * 100.0
            dfnew.loc[dfnew["sp_diff_size"] > 300, "sp_diff_size"] = 300.0
            dfnew.plot.scatter(
                x="longitude",
                y="latitude",
                c=dfnew["sp_diff"],
                s=dfnew["sp_diff_size"],
                vmin=-1 * top,
                vmax=top,
                ax=ax,
                colorbar=True,
                **kwargs,
            )
            if not outline:
                _set_outline_patch_alpha(ax)
            if global_map:
                plt.xlim([-180, 180])
                plt.ylim([-90, 90])
            if tight:
                plt.tight_layout(pad=0)
            return ax
    except ValueError:
        exit