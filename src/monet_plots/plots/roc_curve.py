from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import matplotlib.pyplot as plt
import numpy as np

from ..plot_utils import to_dataframe, validate_dataframe
from ..verification_metrics import compute_auc
from .base import BasePlot

if TYPE_CHECKING:
    import matplotlib.axes


class ROCCurvePlot(BasePlot):
    """
    Receiver Operating Characteristic (ROC) Curve Plot.

    Visualizes the trade-off between Probability of Detection (POD) and
    Probability of False Detection (POFD).

    Functional Requirements:
    1. Plot POD (y-axis) vs POFD (x-axis).
    2. Draw diagonal "no skill" line (0,0) to (1,1).
    3. Calculate and display Area Under Curve (AUC) in legend.
    4. Support multiple models/curves via grouping.

    Edge Cases:
    - Non-monotonic data points (should sort by threshold/prob).
    - Single point provided (cannot calculate AUC properly, return NaN or handle gracefully).
    - Missing columns.
    """

    def __init__(self, fig=None, ax=None, **kwargs):
        super().__init__(fig=fig, ax=ax, **kwargs)

    def plot(
        self,
        data: Any,
        x_col: str = "pofd",
        y_col: str = "pod",
        label_col: Optional[str] = None,
        show_auc: bool = True,
        dim: Optional[str] = None,
        **kwargs: Any,
    ) -> matplotlib.axes.Axes:
        """
        Main plotting method.

        Parameters
        ----------
        data : Any
            Input data containing ROC points. Can be a pandas DataFrame,
            xarray DataArray, xarray Dataset, or numpy ndarray.
        x_col : str, optional
            Variable name for POFD (False Alarm Rate), by default "pofd".
        y_col : str, optional
            Variable name for POD (Hit Rate), by default "pod".
        label_col : str, optional
            Grouping for different curves.
        show_auc : bool, optional
            Whether to calculate and append AUC to labels, by default True.
        dim : str, optional
            Dimension along which to integrate AUC if data is multidimensional Xarray.
        **kwargs : Any
            Additional keyword arguments passed to ax.plot.

        Returns
        -------
        matplotlib.axes.Axes
            The axes object with the ROC curve.
        """
        import xarray as xr

        # Track provenance if Xarray
        if isinstance(data, (xr.DataArray, xr.Dataset)):
            history = data.attrs.get("history", "")
            data.attrs["history"] = f"Generated ROCCurvePlot; {history}"

        # Draw No Skill Line
        self.ax.plot([0, 1], [0, 1], "k--", label="No Skill", alpha=0.5)
        self.ax.grid(True, alpha=0.3)

        if isinstance(data, (xr.DataArray, xr.Dataset)):
            # Native Xarray path
            ds = data
            if label_col:
                labels_concrete = ds[label_col].values
                unique_labels = np.unique(labels_concrete)
                for label in unique_labels:
                    # Select using mask for boolean dask indexing safety
                    mask = labels_concrete == label
                    dim_name = list(ds.dims)[0]
                    ds_sub = ds.isel({dim_name: mask})
                    self._plot_single_curve(
                        ds_sub,
                        x_col,
                        y_col,
                        label=str(label),
                        show_auc=show_auc,
                        dim=dim,
                        **kwargs,
                    )
            else:
                self._plot_single_curve(
                    ds, x_col, y_col, label="Model", show_auc=show_auc, dim=dim, **kwargs
                )
        else:
            # Fallback path
            df = to_dataframe(data)
            validate_dataframe(df, required_columns=[x_col, y_col])

            if label_col:
                groups = df.groupby(label_col)
                for name, group in groups:
                    self._plot_single_curve(
                        group, x_col, y_col, label=str(name), show_auc=show_auc, **kwargs
                    )
            else:
                self._plot_single_curve(
                    df, x_col, y_col, label="Model", show_auc=show_auc, **kwargs
                )

        # Formatting
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_xlabel("Probability of False Detection (POFD)")
        self.ax.set_ylabel("Probability of Detection (POD)")
        self.ax.set_aspect("equal")
        self.ax.legend(loc="lower right")

        return self.ax

    def _plot_single_curve(
        self,
        data: Any,
        x_col: str,
        y_col: str,
        label: str,
        show_auc: bool,
        dim: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Helper to plot a single ROC curve and calculate AUC."""
        import xarray as xr

        if isinstance(data, (xr.DataArray, xr.Dataset)):
            # Xarray path
            ds = data
            # Sort by x_col
            ds_sorted = ds.sortby(x_col)
            x = ds_sorted[x_col]
            y = ds_sorted[y_col]

            auc_str = ""
            if show_auc:
                # compute_auc handles xarray/dask
                auc = compute_auc(x, y, dim=dim)
                auc_val = float(auc.compute()) if hasattr(auc, "compute") else float(auc)
                auc_str = f" (AUC={auc_val:.3f})"

            x_plot = x.values
            y_plot = y.values
        else:
            # Pandas path
            df = data
            df_sorted = df.sort_values(by=x_col).dropna(subset=[x_col, y_col])
            x_plot = df_sorted[x_col].values
            y_plot = df_sorted[y_col].values

            auc_str = ""
            if len(x_plot) >= 2 and show_auc:
                auc = compute_auc(x_plot, y_plot)
                auc_str = f" (AUC={auc:.3f})"

        full_label = label + auc_str
        self.ax.plot(x_plot, y_plot, label=full_label, **kwargs)
        self.ax.fill_between(x_plot, 0, y_plot, alpha=0.2, **kwargs)


# TDD Anchors (Unit Tests):
# 1. test_auc_calculation: Provide points for a known square/triangle, verify AUC.
# 2. test_sorting_order: Provide unsorted ROC points, ensure plot is monotonic.
# 3. test_single_point_auc: Handle case where only 1 threshold point is provided.
