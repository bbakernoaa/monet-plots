from __future__ import annotations

import typing as t

import numpy as np
from matplotlib import pyplot as plt

from .base import BasePlot


class ProfilePlot(BasePlot):
    """Profile or cross-section plot."""

    def __init__(
        self,
        *,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray | None = None,
        **kwargs: t.Any,
    ) -> None:
        """
        Parameters
        ----------
        x
            X-axis data.
        y
            Y-axis data.
        z
            Optional Z-axis data for contour plots.
        **kwargs
            Keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self.x = x
        self.y = y
        self.z = z

    def plot(self, **kwargs: t.Any) -> None:
        """
        Parameters
        ----------
        **kwargs
            Keyword arguments passed to `matplotlib.pyplot.plot` or
            `matplotlib.pyplot.contourf`.
        """
        if self.ax is None:
            if self.fig is None:
                self.fig = plt.figure()
            self.ax = self.fig.add_subplot()

        if self.z is not None:
            self.ax.contourf(self.x, self.y, self.z, **kwargs)
        else:
            self.ax.plot(self.x, self.y, **kwargs)
