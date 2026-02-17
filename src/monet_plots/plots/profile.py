from __future__ import annotations

import typing as t

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from .. import tools
from .base import BasePlot


class ProfilePlot(BasePlot):
    """Profile or cross-section plot."""

    def __init__(
        self,
        *,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray | None = None,
        alt_adjust: float | None = None,
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
        alt_adjust
            Value to subtract from the y-axis data for altitude adjustment.
        **kwargs
            Keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self.x = x
        if alt_adjust is not None:
            self.y = y - alt_adjust
        else:
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

    def hvplot(self, **kwargs: t.Any):
        """Generate an interactive profile plot using hvPlot."""
        import hvplot.pandas  # noqa: F401
        import xarray as xr

        if self.z is not None:
            import hvplot.xarray  # noqa: F401

            da = xr.DataArray(
                self.z, coords={"y": self.y, "x": self.x}, dims=["y", "x"]
            )
            return da.hvplot.contourf(x="x", y="y", title="Profile Contour")
        else:
            df = pd.DataFrame({"x": self.x, "y": self.y})
            return df.hvplot.line(x="x", y="y", title="Profile Plot")


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

    def hvplot(self, **kwargs: t.Any):
        """Generate an interactive vertical slice plot using hvPlot."""
        import hvplot.xarray  # noqa: F401
        import xarray as xr

        da = xr.DataArray(self.z, coords={"y": self.y, "x": self.x}, dims=["y", "x"])
        return da.hvplot.contourf(x="x", y="y", title="Vertical Slice")


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

    def hvplot(self, **kwargs: t.Any):
        """Generate an interactive stick plot using hvPlot."""
        import hvplot.pandas  # noqa: F401

        df = pd.DataFrame({"x": self.x, "y": self.y, "u": self.u, "v": self.v})
        df["angle"] = np.arctan2(df["v"], df["u"])
        df["mag"] = np.sqrt(df["u"] ** 2 + df["v"] ** 2)

        return df.hvplot.vectorfield(
            x="x", y="y", angle="angle", mag="mag", title="Stick Plot"
        )


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
            output_list, vert=False, positions=position_list_mid, **kwargs
        )

    def hvplot(self, **kwargs: t.Any):
        """Generate an interactive vertical box plot using hvPlot."""
        import hvplot.pandas  # noqa: F401

        output_list = tools.split_by_threshold(self.data, self.y, self.thresholds)
        position_list_1 = self.thresholds[:-1]
        position_list_2 = self.thresholds[1:]
        position_list_mid = [
            (p1 + p2) / 2 for p1, p2 in zip(position_list_1, position_list_2)
        ]

        data_rows = []
        for i, vals in enumerate(output_list):
            for v in vals:
                data_rows.append({"value": v, "position": position_list_mid[i]})

        df = pd.DataFrame(data_rows)
        return df.hvplot.box(
            y="value", by="position", invert=True, title="Vertical Box Plot"
        )
