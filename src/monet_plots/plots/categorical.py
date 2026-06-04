import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr


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

    # Only facet when explicitly requested. Using axis-level plotting for the
    # common single-panel case avoids seaborn FacetGrid attachment issues seen
    # in some CI environments.
    facet_col = kwargs.pop("col", None)
    facet_row = kwargs.pop("row", None)
    facet_col_wrap = kwargs.pop("col_wrap", col_wrap)
    should_facet = facet_col is not None or facet_row is not None

    with plt.style.context(style.wiley_style):
        if should_facet:
            catplot_kwargs = {
                "data": df,
                "kind": kind,
                "sharey": sharey,
                "col": facet_col,
                "row": facet_row,
                "col_wrap": facet_col_wrap,
                **kwargs,
            }
            p = sns.catplot(**catplot_kwargs)
            p.fig.set_size_inches(figsize)
            fig = p.fig
            axes = p.axes
            first_ax = axes.flatten()[0]
        else:
            fig, ax = plt.subplots(figsize=figsize)
            x = kwargs.pop("x")
            y = kwargs.pop("y")

            if kind == "bar":
                sns.barplot(data=df, x=x, y=y, ax=ax, **kwargs)
            elif kind == "violin":
                sns.violinplot(data=df, x=x, y=y, ax=ax, **kwargs)
            else:
                raise ValueError("kind must be 'bar' or 'violin'")

            axes = np.array([[ax]])
            first_ax = ax

        if title is not None:
            fig.suptitle(title)

        if ylabel is not None:
            first_ax.set_ylabel(ylabel)
        if xlabel is not None:
            first_ax.set_xlabel(xlabel)

        if legend == "auto":
            legend = "hue" in kwargs

        if legend is True:
            if isinstance(legend_labels, list) and len(legend_labels) > 0:
                # To be implemented: custom legend labels
                pass
            if not should_facet:
                handles, labels = first_ax.get_legend_handles_labels()
                if handles:
                    fig.legend(handles, labels)

    return fig, axes


def categorical_timeseries(data, **kwargs):
    """Make a timeseries of categorical plots.

    (Placeholder for future implementation)
    """
    raise NotImplementedError("Categorical timeseries plots are not yet implemented.")
