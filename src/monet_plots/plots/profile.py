from __future__ import annotations

import typing as t

import numpy as np
from matplotlib import pyplot as plt

from .. import tools
from .base import BasePlot


class ProfilePlot(BasePlot):
    """
    Profile or cross-section plot (Unified API).

    Parameters
    ----------
    data : Any, optional
        Not used, for API consistency.
    var1 : np.ndarray
        X-axis data.
    var2 : np.ndarray
        Y-axis data.
    z : np.ndarray, optional
        Optional Z-axis data for contour plots.
    alt_adjust : float, optional
        Value to subtract from the y-axis data for altitude adjustment.
    x : np.ndarray, optional
        Legacy alias for var1.
    y : np.ndarray, optional
        Legacy alias for var2.
    **kwargs : Any
        Additional keyword arguments.
    """

    def __init__(
        self,
        *,
        data: t.Any = None,
        var1: np.ndarray = None,
        var2: np.ndarray = None,
        z: np.ndarray | None = None,
        alt_adjust: float | None = None,
        x: np.ndarray = None,  # legacy alias
        y: np.ndarray = None,  # legacy alias
        **kwargs: t.Any,
    ) -> None:
        super().__init__(**kwargs)
        self.x = var1 if var1 is not None else x
        if alt_adjust is not None:
            self.y = (var2 if var2 is not None else y) - alt_adjust
        else:
            self.y = var2 if var2 is not None else y
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


class VerticalSlice(ProfilePlot):
    """Vertical cross-section plot."""

    def __init__(self, *args, **kwargs):
        """
        Initialize the vertical slice plot.
        """
        super().__init__(*args, **kwargs)

    def plot(self, **kwargs: t.Any) -> None:
        """
        Parameters
        ----------
        **kwargs
            Keyword arguments passed to `matplotlib.pyplot.contourf`.
        """
        if self.ax is None:
            if self.fig is None:
                self.fig = plt.figure()
            self.ax = self.fig.add_subplot()

        self.ax.contourf(self.x, self.y, self.z, **kwargs)


class StickPlot(BasePlot):
    """Vertical stick plot."""

    def __init__(self, u, v, y, *args, **kwargs):
        """
        Initialize the stick plot.
        Args:
            u (np.ndarray, pd.Series, xr.DataArray): U-component of the vector.
            v (np.ndarray, pd.Series, xr.DataArray): V-component of the vector.
            y (np.ndarray, pd.Series, xr.DataArray): Vertical coordinate.
            **kwargs: Additional keyword arguments passed to BasePlot.
        """
        super().__init__(*args, **kwargs)
        self.u = u
        self.v = v
        self.y = y
        self.x = np.zeros_like(self.y)

    def plot(self, **kwargs: t.Any) -> None:
        """
        Parameters
        ----------
        **kwargs
            Keyword arguments passed to `matplotlib.pyplot.quiver`.
        """
        if self.ax is None:
            if self.fig is None:
                self.fig = plt.figure()
            self.ax = self.fig.add_subplot()

        return self.ax.quiver(self.x, self.y, self.u, self.v, **kwargs)


class VerticalBoxPlot(BasePlot):
    """Vertical box plot."""

    def __init__(self, data, y, thresholds, *args, **kwargs):
        """
        Initialize the vertical box plot.
        Args:
            data (np.ndarray, pd.Series, xr.DataArray): Data to plot.
            y (np.ndarray, pd.Series, xr.DataArray): Vertical coordinate.
            thresholds (list): List of thresholds to bin the data.
            **kwargs: Additional keyword arguments passed to BasePlot.
        """
        super().__init__(*args, **kwargs)
        self.data = data
        self.y = y
        self.thresholds = thresholds

    def plot(self, **kwargs: t.Any) -> None:
        """
        Parameters
        ----------
        **kwargs
            Keyword arguments passed to `matplotlib.pyplot.boxplot`.
        """
        if self.ax is None:
            if self.fig is None:
                self.fig = plt.figure()
            self.ax = self.fig.add_subplot()

        output_list = tools.split_by_threshold(self.data, self.y, self.thresholds)
        position_list_1 = self.thresholds[:-1]
        position_list_2 = self.thresholds[1:]
        position_list_mid = [
            (p1 + p2) / 2 for p1, p2 in zip(position_list_1, position_list_2)
        ]

        return self.ax.boxplot(
            output_list, orientation="horizontal", positions=position_list_mid, **kwargs
        )
