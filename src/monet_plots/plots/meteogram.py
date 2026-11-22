from __future__ import annotations

import typing as t

import pandas as pd
from ..plot_utils import set_concise_time_axis
from .base import BasePlot, figure_style


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
        self,
        *,
        data: pd.DataFrame = None,
        var1: list[str] = None,
        df: pd.DataFrame = None,
        variables: list[str] = None,
        **kwargs: t.Any,
    ) -> None:
        owns_figure = kwargs.get("fig") is None and kwargs.get("ax") is None
        super().__init__(**kwargs)
        if owns_figure:
            self.ax.remove()
            self.ax = None
        self._panels = []
        if df is not None and data is None:
            data = df
        self.data = data
        self.var1 = var1 or variables

    @figure_style
    def plot(self, **kwargs: t.Any) -> None:
        """
        Parameters
        ----------
        **kwargs
            Keyword arguments passed to `matplotlib.pyplot.plot`.
        """
        if not self.var1:
            raise ValueError("Choose at least one variable for the meteogram.")
        for panel in self._panels:
            panel.remove()
        self._panels = []
        n_vars = len(self.var1)
        for i, var in enumerate(self.var1):
            shared = self._panels[0] if self._panels else None
            ax = self.fig.add_subplot(n_vars, 1, i + 1, sharex=shared)
            ax.plot(self.data.index, self.data[var], **kwargs)
            ax.set_ylabel(var)
            ax.tick_params(axis="x", labelbottom=i == n_vars - 1)
            self._panels.append(ax)
        self.ax = self._panels[:]
        if isinstance(self.data.index, pd.DatetimeIndex):
            timezone = self.data.index.tz
            set_concise_time_axis(self.ax[-1], timezone)
            self.ax[-1].set_xlabel(
                f"Valid time ({timezone})" if timezone else "Valid time"
            )
