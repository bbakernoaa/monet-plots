from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, Union

import numpy as np
import xarray as xr

from .base import BasePlot

if TYPE_CHECKING:
    import matplotlib.axes
from ..plot_utils import normalize_data, validate_dataframe
from ..verification_metrics import (
    _update_history,
    compute_categorical_metrics,
    compute_pod,
    compute_success_ratio,
)


class PerformanceDiagramPlot(BasePlot):
    """
    Performance Diagram Plot (Roebber).

    Visualizes the relationship between Probability of Detection (POD),
    Success Ratio (SR), Critical Success Index (CSI), and Bias.

    Supports lazy evaluation via Xarray and Dask for large datasets.
    """

    def __init__(self, fig=None, ax=None, **kwargs):
        super().__init__(fig=fig, ax=ax, **kwargs)

    def plot(
        self,
        data: Any,
        x_col: str = "success_ratio",
        y_col: str = "pod",
        counts_cols: Optional[List[str]] = None,
        obs_col: Optional[str] = None,
        mod_col: Optional[str] = None,
        threshold: Optional[float] = None,
        label_col: Optional[str] = None,
        dim: Optional[Union[str, List[str]]] = None,
        **kwargs: Any,
    ) -> matplotlib.axes.Axes:
        """
        Generate the Performance Diagram.

        Parameters
        ----------
        data : Any
            Input data. Can be a pandas DataFrame, xarray DataArray,
            xarray Dataset, or numpy ndarray.
        x_col : str, default 'success_ratio'
            Column or variable name for Success Ratio (1-FAR).
        y_col : str, default 'pod'
            Column or variable name for Probability of Detection (POD).
        counts_cols : list, optional
            List of columns/variables [hits, misses, fa, cn] to calculate
            metrics if x_col and y_col are not present.
        obs_col : str, optional
            Variable name for observations. Required if `threshold` is used.
        mod_col : str, optional
            Variable name for model values. Required if `threshold` is used.
        threshold : float, optional
            Threshold for converting raw data to categorical events.
        label_col : str, optional
            Column or coordinate to use for legend labels and grouping.
        dim : str or list of str, optional
            Dimension(s) to reduce over when calculating metrics from raw data.
        **kwargs : Any
            Keyword arguments passed to `ax.plot` for the data points.
        """
        data = normalize_data(data)

        # Track provenance
        if isinstance(data, (xr.DataArray, xr.Dataset)):
            _update_history(data, "Plotted with PerformanceDiagramPlot")

        # Data Preparation
        if threshold is not None and obs_col and mod_col:
            m = compute_categorical_metrics(
                data[obs_col], data[mod_col], threshold, dim=dim
            )
            df_plot = xr.Dataset({x_col: m["success_ratio"], y_col: m["pod"]})
            # Try to preserve label_col if it's still there after reduction
            if label_col and label_col in data.coords:
                df_plot = df_plot.assign_coords({label_col: data[label_col]})
        else:
            # Traditional counts or pre-calculated metrics
            if not isinstance(data, (xr.DataArray, xr.Dataset)):
                self._validate_inputs(data, x_col, y_col, counts_cols)
            df_plot = self._prepare_data(data, x_col, y_col, counts_cols)

        # Plot Background (Isolines)
        self._draw_background()

        # Plot Data
        if label_col:
            # Handle both Xarray and Pandas grouping
            if isinstance(df_plot, (xr.DataArray, xr.Dataset)):
                groups = df_plot.groupby(label_col)
                for name, group in groups:
                    self.ax.plot(
                        group[x_col],
                        group[y_col],
                        marker="o",
                        label=name,
                        linestyle="none",
                        **kwargs,
                    )
            else:
                for name, group in df_plot.groupby(label_col):
                    self.ax.plot(
                        group[x_col],
                        group[y_col],
                        marker="o",
                        label=name,
                        linestyle="none",
                        **kwargs,
                    )
            self.ax.legend(loc="best")
        else:
            self.ax.plot(
                df_plot[x_col], df_plot[y_col], marker="o", linestyle="none", **kwargs
            )

        # Formatting
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_xlabel("Success Ratio (1-FAR)")
        self.ax.set_ylabel("Probability of Detection (POD)")
        self.ax.set_aspect("equal")
        return self.ax

    def _validate_inputs(self, data, x, y, counts):
        """Validates input dataframe structure."""
        if counts:
            validate_dataframe(data, required_columns=counts)
        else:
            validate_dataframe(data, required_columns=[x, y])

    def _prepare_data(self, data, x, y, counts):
        """
        Calculates metrics if counts provided, otherwise returns subset.
        """
        if counts:
            hits_col, misses_col, fa_col, cn_col = counts
            if isinstance(data, (xr.DataArray, xr.Dataset)):
                # Vectorized Xarray calculation (lazy if Dask-backed)
                res_x = compute_success_ratio(data[hits_col], data[fa_col])
                res_y = compute_pod(data[hits_col], data[misses_col])
                if isinstance(data, xr.Dataset):
                    data = data.assign({x: res_x, y: res_y})
                else:
                    data = xr.Dataset({x: res_x, y: res_y})
            else:
                # Pandas calculation
                data = data.copy()
                data[x] = compute_success_ratio(data[hits_col], data[fa_col])
                data[y] = compute_pod(data[hits_col], data[misses_col])
        return data

    def _draw_background(self):
        """
        Draws CSI and Bias isolines.

        Pseudocode:
        1. Create meshgrid for x (SR) and y (POD) from 0.01 to 1.
        2. Calculate CSI = 1 / (1/SR + 1/POD - 1).
        3. Calculate Bias = POD / SR.
        4. Contour plot CSI (dashed).
        5. Contour plot Bias (dotted).
        6. Label contours.
        """
        # Avoid division by zero at boundaries
        xx, yy = np.meshgrid(np.linspace(0.01, 0.99, 50), np.linspace(0.01, 0.99, 50))
        csi = (xx * yy) / (xx + yy - xx * yy)
        bias = yy / xx

        # CSI contours (dashed, lightgray)
        cs_csi = self.ax.contour(
            xx,
            yy,
            csi,
            levels=np.arange(0.1, 0.95, 0.1),
            colors="lightgray",
            linestyles="--",
            alpha=0.6,
        )
        self.ax.clabel(cs_csi, inline=True, fontsize=8, fmt="%.1f")

        # Bias contours (dotted, darkgray)
        cs_bias = self.ax.contour(
            xx,
            yy,
            bias,
            levels=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
            colors="darkgray",
            linestyles=":",
            alpha=0.6,
        )
        self.ax.clabel(cs_bias, inline=True, fontsize=8, fmt="%.1f")

        # Perfect forecast line
        self.ax.plot([0.01, 0.99], [0.01, 0.99], "k-", linewidth=1.5, alpha=0.8)

        # TDD Anchor: Test that contours are within 0-1 range.


# TDD Anchors (Unit Tests):
# 1. test_metric_calculation_from_counts: Provide hits/misses/fa, verify SR/POD output.
# 2. test_perfect_score_location: Ensure perfect forecast plots at (1,1).
# 3. test_missing_columns_error: Assert ValueError if cols missing.
# 4. test_background_drawing: Mock plt.contour, verify calls with correct grids.
