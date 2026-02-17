from __future__ import annotations

import typing as t

import numpy as np
import pandas as pd

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
        if "subplot_kw" not in kwargs:
            kwargs["subplot_kw"] = {"projection": "polar"}
        elif "projection" not in kwargs["subplot_kw"]:
            kwargs["subplot_kw"]["projection"] = "polar"

        super().__init__(**kwargs)

        if self.ax is None:
            self.ax = self.fig.add_subplot(projection="polar")

        from matplotlib.projections.polar import PolarAxes

        if not isinstance(self.ax, PolarAxes):
            raise ValueError("Windrose plot requires a polar axis.")

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
        if isinstance(bins, int):
            bins = np.linspace(0, 360, bins + 1)
        if isinstance(rose_bins, int):
            rose_bins = np.linspace(self.ws.min(), self.ws.max(), rose_bins + 1)

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

    def hvplot(self, **kwargs: t.Any):
        """Generate an interactive windrose-like plot using hvPlot."""
        import hvplot.pandas  # noqa: F401

        df = pd.DataFrame({"ws": self.ws, "wd": self.wd})

        # Simplified interactive windrose: scatter on polar plot
        # Note: hvPlot doesn't have a native polar bar plot yet.
        plot_kwargs = {
            "x": "wd",
            "y": "ws",
            "kind": "scatter",
            "title": "Interactive Windrose (Simplified)",
        }
        plot_kwargs.update(kwargs)

        return df.hvplot(**plot_kwargs)
