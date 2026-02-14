import seaborn as sns
import xarray as xr
import matplotlib.pyplot as plt


def categorical_plot(
    data,
    *,
    kind="bar",
    col_wrap=3,
    figsize=(15, 8),
    title=None,
    legend="auto",
    legend_labels=None,
    xlabel=None,
    ylabel=None,
    sharey=True,
    **kwargs,
):
    """Make a categorical plot (bar or violin).

    Parameters
    ----------
    data : xarray.Dataset or xarray.DataArray
        Data to be plotted.
    kind : str, optional
        Type of plot, 'bar' or 'violin', by default 'bar'.
    col_wrap : int, optional
        Number of columns for subplot grid, by default 3.
    figsize : tuple, optional
        Figure size, by default (15, 8).
    title : str, optional
        Plot title, by default None.
    legend : str, optional
        Legend type, by default "auto".
    legend_labels : list of str, optional
        Legend labels, by default None.
    xlabel : str, optional
        X-axis label, by default None.
    ylabel : str, optional
        Y-axis label, by default None.
    sharey : bool, optional
        Whether to share the y-axis, by default True.
    **kwargs
        Additional keyword arguments passed to seaborn.catplot().

    Returns
    -------
    fig, ax
        Matplotlib figure and axes objects.
    """
    if "x" not in kwargs or "y" not in kwargs:
        raise ValueError("x and y must be provided as keyword arguments")

    if isinstance(data, xr.DataArray) and data.name is None:
        data.name = kwargs["y"]

    df = data.to_dataframe().reset_index()

    from .. import style

    col = "site" if "site" in df else None
    with plt.style.context(style.wiley_style):
        p = sns.catplot(
            data=df,
            kind=kind,
            col=col,
            col_wrap=col_wrap if col is not None else None,
            sharey=sharey,
            **kwargs,
        )
        p.fig.set_size_inches(figsize)

        if title is not None:
            p.fig.suptitle(title)

        if ylabel is not None:
            p.axes.flatten()[0].set_ylabel(ylabel)
        if xlabel is not None:
            p.axes.flatten()[0].set_xlabel(xlabel)

        if legend == "auto":
            if "hue" in kwargs:
                legend = True
            else:
                legend = False

        if legend is True:
            if isinstance(legend_labels, list) and len(legend_labels) > 0:
                # To be implemented: custom legend labels
                pass
            # Seaborn handles legend automatically when using 'hue'
            pass

    return p.fig, p.axes


def categorical_timeseries(data, **kwargs):
    """Make a timeseries of categorical plots.

    (Placeholder for future implementation)
    """
    raise NotImplementedError("Categorical timeseries plots are not yet implemented.")


def categorical_hvplot(
    data,
    *,
    kind="bar",
    title=None,
    xlabel=None,
    ylabel=None,
    **kwargs,
):
    """Make an interactive categorical plot using hvPlot.

    Parameters
    ----------
    data : xarray.Dataset or xarray.DataArray
        Data to be plotted.
    kind : str, optional
        Type of plot, 'bar' or 'violin', by default 'bar'.
    title : str, optional
        Plot title, by default None.
    xlabel : str, optional
        X-axis label, by default None.
    ylabel : str, optional
        Y-axis label, by default None.
    **kwargs
        Additional keyword arguments passed to hvplot().

    Returns
        A HoloViews object.
    """
    import hvplot.pandas  # noqa: F401
    import hvplot.xarray  # noqa: F401

    if "x" not in kwargs or "y" not in kwargs:
        raise ValueError("x and y must be provided as keyword arguments")

    plot_kwargs = {
        "kind": kind,
        "title": title,
        "xlabel": xlabel,
        "ylabel": ylabel,
    }
    plot_kwargs.update(kwargs)

    return data.hvplot(**plot_kwargs)
