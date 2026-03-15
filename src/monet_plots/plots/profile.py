from __future__ import annotations

import typing as t

import numpy as np
from matplotlib import pyplot as plt
import xarray as xr

from .. import tools, verification_metrics
from ..plot_utils import _update_history, normalize_data
from .base import BasePlot


class ProfilePlot(BasePlot):
    """Profile or cross-section plot."""

    def __init__(
        self,
        *,
        x: np.ndarray | xr.DataArray,
        y: np.ndarray | xr.DataArray,
        z: np.ndarray | xr.DataArray | None = None,
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


class VerticalProfilePlot(BasePlot):
    """Vertical profile plot comparing observations and models.

    Supports multiple models, binned statistics (median, IQR, box-whisker),
    and both shading and boxplot styles.
    """

    def __init__(
        self,
        data: Any,
        obs_col: str,
        mod_cols: Union[str, list[str]],
        alt_col: str = "altitude",
        *,
        bins: Any = 10,
        interquartile_style: str = "shading",
        fig: plt.Figure | None = None,
        ax: plt.Axes | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the vertical profile plot.

        Parameters
        ----------
        data : Any
            Input data (Pandas DataFrame, Xarray Dataset/DataArray).
        obs_col : str
            Column name for observations.
        mod_cols : Union[str, list[str]]
            Column name(s) for model data.
        alt_col : str, optional
            Column name for altitude/pressure, by default "altitude".
        bins : Any, optional
            Number of bins or bin edges for vertical grouping, by default 10.
        interquartile_style : str, optional
            Style for representing the interquartile range: 'shading' or 'box',
            by default 'shading'.
        fig : plt.Figure, optional
            Existing figure object, by default None.
        ax : plt.Axes, optional
            Existing axes object, by default None.
        **kwargs : Any
            Additional arguments passed to BasePlot.
        """
        super().__init__(fig=fig, ax=ax, **kwargs)
        self.data = normalize_data(data)
        self.obs_col = obs_col
        self.mod_cols = [mod_cols] if isinstance(mod_cols, str) else mod_cols
        self.alt_col = alt_col
        self.bins = bins
        self.interquartile_style = interquartile_style

        _update_history(self.data, "Initialized monet-plots.VerticalProfilePlot")

    def plot(self, **kwargs: Any) -> plt.Axes:
        """
        Generate the vertical profile plot.

        Parameters
        ----------
        **kwargs : Any
            Keyword arguments passed to `matplotlib.pyplot.plot` or `ax.bxp`.
        """
        if self.ax is None:
            self.ax = self.fig.add_subplot(1, 1, 1)

        all_cols = [self.obs_col, self.alt_col] + self.mod_cols
        # For Xarray, we might have dimensions to reduce.
        # If it's aircraft/sonde data, it's often 1D already.
        # But we'll handle the grouping generally.

        # Calculate statistics for each variable
        for i, col in enumerate([self.obs_col] + self.mod_cols):
            stats = verification_metrics.compute_binned_quantiles(
                self.data[col],
                self.data[self.alt_col],
                bins=self.bins,
                quantiles=[0.1, 0.25, 0.5, 0.75, 0.9],
            )

            # Performance: compute only once
            if hasattr(stats, "compute"):
                stats = stats.compute()

            label = col
            color = kwargs.get("color", plt.cm.tab10(i))

            if self.interquartile_style == "shading":
                # Plot median line
                self.ax.plot(
                    stats["q0.5"],
                    stats["bin_center"],
                    label=f"{label} (median)",
                    color=color,
                    **kwargs,
                )
                # Plot IQR shading
                self.ax.fill_betweenx(
                    stats["bin_center"],
                    stats["q0.25"],
                    stats["q0.75"],
                    alpha=0.3,
                    color=color,
                )
            elif self.interquartile_style == "box":
                # Create boxplot data for ax.bxp
                bxp_stats = []
                for j in range(len(stats["bin_center"])):
                    # Check if we have valid data in this bin
                    if np.isnan(stats["q0.5"].values[j]):
                        continue

                    bxp_stats.append(
                        {
                            "med": stats["q0.5"].values[j],
                            "q1": stats["q0.25"].values[j],
                            "q3": stats["q0.75"].values[j],
                            "whislo": stats["q0.1"].values[j],
                            "whishi": stats["q0.9"].values[j],
                            "fliers": [],
                        }
                    )

                if bxp_stats:
                    positions = stats["bin_center"].values[
                        ~np.isnan(stats["q0.5"].values)
                    ]
                    # Determine width based on bin size
                    width = (
                        np.diff(stats["bin_center"].values).min() * 0.5
                        if len(positions) > 1
                        else 0.5
                    )

                    self.ax.bxp(
                        bxp_stats,
                        vert=False,
                        positions=positions,
                        widths=width,
                        boxprops=dict(color=color),
                        whiskerprops=dict(color=color),
                        capprops=dict(color=color),
                        medianprops=dict(color=color, linewidth=2),
                        showfliers=False,
                    )
                    # Also plot a dummy line for the legend
                    self.ax.plot([], [], color=color, label=f"{label} (box-whisker)")

        self.ax.set_ylabel(self.alt_col)
        self.ax.set_xlabel(self.obs_col)
        self.ax.legend()

        return self.ax

    def hvplot(self, **kwargs: Any) -> Any:
        """Generate an interactive vertical profile plot using hvPlot (Track B)."""
        try:
            import hvplot.xarray  # noqa: F401
        except ImportError:
            raise ImportError(
                "hvplot is required for interactive plotting. Install it with 'pip install hvplot'."
            )

        # Basic implementation: median line only for interactive
        plots = []
        for i, col in enumerate([self.obs_col] + self.mod_cols):
            stats = verification_metrics.compute_binned_quantiles(
                self.data[col],
                self.data[self.alt_col],
                bins=self.bins,
                quantiles=[0.5],
            )
            plots.append(
                stats.hvplot.line(x="q0.5", y="bin_center", label=col, **kwargs)
            )

        import holoviews as hv

        return hv.Overlay(plots).opts(title="Interactive Vertical Profile")


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
            output_list, vert=False, positions=position_list_mid, **kwargs
        )
