from __future__ import annotations

import typing as t

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from .base import BasePlot


class Windrose(BasePlot):
    """Windrose plot."""

    def __init__(self, *, wd: np.ndarray, ws: np.ndarray, **kwargs: t.Any) -> None:
        """
        Parameters
        ----------
        wd
            wind direction
        ws
            wind speed
        **kwargs
            Keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self.wd = wd
        self.ws = ws

    def plot(
        self,
        *,
        bins: int | np.ndarray = 16,
        rose_bins: int | np.ndarray = 5,
        **kwargs: t.Any,
    ) -> None:
        """
        Parameters
        ----------
        bins
            Number of bins for the wind direction.
        rose_bins
            Number of bins for the wind speed.
        **kwargs
            Keyword arguments passed to `matplotlib.pyplot.bar`.
        """
        if self.fig is None:
            self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection="polar")

        if isinstance(bins, int):
            bins = np.linspace(0, 360, bins + 1)
        if isinstance(rose_bins, int):
            rose_bins = np.linspace(self.ws.min(), self.ws.max(),
                                    rose_bins + 1)

        rose = pd.cut(self.ws, bins=rose_bins, right=True, include_lowest=True)
        angle = pd.cut(
            self.wd,
            bins=bins,
            right=False,
            include_lowest=True,
            labels=np.deg2rad(bins[:-1]),
        )

        ct = pd.crosstab(rose, angle, normalize="all")
        ct = ct.cumsum()

        self.ax.set_theta_zero_location("N")
        self.ax.set_theta_direction(-1)
        self.ax.set_rgrids(
            np.linspace(0.2, 1.0, 5) * ct.iloc[-1].max(),
            labels=np.round(np.linspace(0.2, 1.0, 5) * 100).astype(int),
            angle=180,
        )

        for i in reversed(ct.index):
            self.ax.bar(
                ct.columns.astype(float) + np.pi / bins.size,
                ct.loc[i].values,
                width=2 * np.pi / bins.size,
                label=i,
                **kwargs,
            )

        self.ax.legend()
