from __future__ import annotations

import typing as t

import pandas as pd
from matplotlib import pyplot as plt

from .base import BasePlot


class Meteogram(BasePlot):
    """
    Meteogram plot (Unified API).

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with time series data.
    var1 : list[str]
        List of variables to plot.
    df : pd.DataFrame, optional
        Deprecated alias for ``data``.
    variables : list[str], optional
        Legacy alias for var1.
    **kwargs : Any
        Additional keyword arguments.
    """

    def __init__(
        self, *, data: pd.DataFrame = None, var1: list[str] = None, df: pd.DataFrame = None, variables: list[str] = None, **kwargs: t.Any
    ) -> None:
        if "fig" not in kwargs and "ax" not in kwargs:
            kwargs["fig"] = plt.figure()
        super().__init__(**kwargs)
        if df is not None and data is None:
            data = df
        self.data = data
        self.var1 = var1 or variables

    def plot(self, **kwargs: t.Any) -> None:
        """
        Parameters
        ----------
        **kwargs
            Keyword arguments passed to `matplotlib.pyplot.plot`.
        """
        if self.fig is None:
            self.fig = plt.figure()

        n_vars = len(self.var1)
        for i, var in enumerate(self.var1):
            ax = self.fig.add_subplot(n_vars, 1, i + 1)
            ax.plot(self.data.index, self.data[var], **kwargs)
            ax.set_ylabel(var)
            if i < n_vars - 1:
                plt.setp(ax.get_xticklabels(), visible=False)

        self.ax = self.fig.get_axes()
