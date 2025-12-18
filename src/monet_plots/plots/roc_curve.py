import numpy as np
import pandas as pd
from typing import Optional, Any
from .base import BasePlot
from ..plot_utils import validate_dataframe, to_dataframe
from ..verification_metrics import compute_auc


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
        **kwargs,
    ):
        """
        Main plotting method.

        Args:
            data (pd.DataFrame, np.ndarray, xr.Dataset, xr.DataArray): Input data containing ROC points.
            x_col (str): Column name for POFD (False Alarm Rate).
            y_col (str): Column name for POD (Hit Rate).
            label_col (str, optional): Column for grouping different curves.
            show_auc (bool): Whether to calculate and append AUC to labels.
            **kwargs: Matplotlib kwargs.
        """
        df = to_dataframe(data)
        # TDD Anchor: Test validation raises error on missing cols
        validate_dataframe(df, required_columns=[x_col, y_col])

        # Draw No Skill Line
        self.ax.plot([0, 1], [0, 1], "k--", label="No Skill", alpha=0.5)
        self.ax.grid(True, alpha=0.3)

        if label_col:
            groups = df.groupby(label_col)
            for name, group in groups:
                self._plot_single_curve(group, x_col, y_col, label=str(name), show_auc=show_auc, **kwargs)
            self.ax.legend(loc="lower right")
        else:
            self._plot_single_curve(df, x_col, y_col, label="Model", show_auc=show_auc, **kwargs)

        # Formatting
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_xlabel("Probability of False Detection (POFD)")
        self.ax.set_ylabel("Probability of Detection (POD)")
        self.ax.set_aspect("equal")

    def _plot_single_curve(self, df, x_col, y_col, label, show_auc, **kwargs):
        """
        Helper to plot a single ROC curve and calc AUC.

        Pseudocode:
        1. Sort df by x_col (POFD) ascending.
        2. Get x (POFD) and y (POD) arrays.
        3. If show_auc:
            auc = trapz(y, x)
            label += f" (AUC={auc:.3f})"
        4. self.ax.plot(x, y, label=label, **kwargs)
        """
        # TDD Anchor: Test AUC calculation against sklearn.metrics.auc or manual known values.
        # TDD Anchor: Ensure sorting is applied correctly.

        df_sorted = df.sort_values(by=x_col).dropna(subset=[x_col, y_col])
        x = df_sorted[x_col].values
        y = df_sorted[y_col].values

        auc_str = ""
        if len(x) >= 2 and show_auc:
            auc = compute_auc(x, y)
            auc_str = f" (AUC={auc:.3f})"

        full_label = label + auc_str
        self.ax.plot(x, y, label=full_label, **kwargs)
        self.ax.fill_between(x, 0, y, alpha=0.2, **kwargs)


# TDD Anchors (Unit Tests):
# 1. test_auc_calculation: Provide points for a known square/triangle, verify AUC.
# 2. test_sorting_order: Provide unsorted ROC points, ensure plot is monotonic.
# 3. test_single_point_auc: Handle case where only 1 threshold point is provided.
