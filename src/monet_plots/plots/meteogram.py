from __future__ import annotations

import typing as t

import pandas as pd
from matplotlib import pyplot as plt

from .base import BasePlot


class Meteogram(BasePlot):
    """Meteogram plot."""

    def __init__(
        self, *, df: pd.DataFrame, variables: list[str], **kwargs: t.Any
    ) -> None:
        """
        Parameters
        ----------
        df
            DataFrame with time series data.
        variables
            List of variables to plot.
        **kwargs
            Keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self.df = df
        self.variables = variables

    def plot(self, **kwargs: t.Any) -> None:
        """
        Parameters
        ----------
        **kwargs
            Keyword arguments passed to `matplotlib.pyplot.plot`.
        """
        if self.fig is None:
            self.fig = plt.figure()

        n_vars = len(self.variables)
        for i, var in enumerate(self.variables):
            ax = self.fig.add_subplot(n_vars, 1, i + 1)
            ax.plot(self.df.index, self.df[var], **kwargs)
            ax.set_ylabel(var)
            if i < n_vars - 1:
                plt.setp(ax.get_xticklabels(), visible=False)

        self.ax = self.fig.get_axes()
