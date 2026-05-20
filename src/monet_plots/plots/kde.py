# src/monet_plots/plots/kde.py

import seaborn as sns

from .base import BasePlot


class KDEPlot(BasePlot):
    """
    Create a kernel density estimate plot (Unified API).

    This plot shows the distribution of a single variable.

    Parameters
    ----------
    data : Any
        Data to plot.
    var1 : str, optional
        Column name for the variable to plot (x or y).
    title : str, optional
        Title for the plot.
    label : str, optional
        Label for the plot.
    df : Any, optional
        Deprecated alias for ``data``.
    x : str, optional
        Legacy alias for var1.
    y : str, optional
        Legacy alias for var1.
    **kwargs : Any
        Additional keyword arguments.
    """

    def __init__(self, data=None, var1=None, title=None, label=None, *args, df=None, x=None, y=None, **kwargs):
        super().__init__(*args, **kwargs)
        if df is not None and data is None:
            data = df
        self.data = data
        self.var1 = var1 or x or y
        self.title = title
        self.label = label

    def plot(self, **kwargs):
        """Generate the KDE plot."""
        with sns.axes_style("ticks"):
            # If var1 is set, use as x; else fallback to y for 1D, or both for 2D
            if self.var1:
                self.ax = sns.kdeplot(
                    data=self.data, x=self.var1, ax=self.ax, label=self.label, **kwargs
                )
            else:
                self.ax = sns.kdeplot(
                    data=self.data, ax=self.ax, label=self.label, **kwargs
                )
            if self.title:
                self.ax.set_title(self.title)
            sns.despine()
        return self.ax
