import numpy as np
from typing import Optional, List, Any
from ..verification_metrics import compute_rev
from .base import BasePlot
from ..plot_utils import validate_dataframe, to_dataframe


class RelativeEconomicValuePlot(BasePlot):
    """
    Relative Economic Value (REV) Plot.

    Visualizes the potential economic value of a forecast system relative to climatology.

    Functional Requirements:
    1. Plot Value (y-axis) vs Cost/Loss Ratio (x-axis).
    2. Calculate REV based on Hits, Misses, False Alarms, Correct Negatives.
    3. Support multiple models.
    4. X-axis usually logarithmic or specific range [0, 1].

    Edge Cases:
    - C/L ratio 0 or 1 (value is 0).
    - No events observed (metrics undefined).
    """

    def __init__(self, fig=None, ax=None, **kwargs):
        super().__init__(fig=fig, ax=ax, **kwargs)

    def plot(
        self,
        data: Any,
        counts_cols: List[str] = ["hits", "misses", "fa", "cn"],
        climatology: Optional[float] = None,
        label_col: Optional[str] = None,
        cost_loss_ratios: Optional[np.ndarray] = None,
        **kwargs,
    ):
        """
        Main plotting method.

        Args:
            data (pd.DataFrame, np.ndarray, xr.Dataset, xr.DataArray): Input data with contingency table counts.
            counts_cols (List[str]): Contingency table columns [hits, misses, fa, cn].
            climatology (Optional[float]): Sample climatology (base rate). Computed if None.
            label_col (Optional[str]): Grouping column for multiple curves.
            cost_loss_ratios (Optional[np.ndarray]): Array of C/L ratios. Default linspace(0.001,0.999,100).
            **kwargs: Matplotlib kwargs.
        """
        df = to_dataframe(data)
        validate_dataframe(df, required_columns=counts_cols)

        if climatology is None:
            total_events = df[counts_cols[0]].sum() + df[counts_cols[1]].sum()
            total = total_events + df[counts_cols[2]].sum() + df[counts_cols[3]].sum()
            climatology = total_events / total if total > 0 else 0.5

        if cost_loss_ratios is None:
            cost_loss_ratios = np.linspace(0.001, 0.999, 100)

        # TDD Anchor: Test REV calculation logic

        if label_col:
            for name, group in df.groupby(label_col):
                rev_values = self._calculate_rev(
                    group, counts_cols, cost_loss_ratios, climatology
                )
                self.ax.plot(cost_loss_ratios, rev_values, label=str(name), **kwargs)
            self.ax.legend(loc="best")
        else:
            rev_values = self._calculate_rev(
                df, counts_cols, cost_loss_ratios, climatology
            )
            self.ax.plot(cost_loss_ratios, rev_values, label="Model", **kwargs)

        self.ax.set_xlabel("Cost/Loss Ratio")
        self.ax.set_ylabel("Relative Economic Value (REV)")
        self.ax.set_ylim(-0.2, 1.05)
        self.ax.axhline(0, color="k", linestyle="--", alpha=0.7, label="Climatology")
        self.ax.axhline(1, color="gray", linestyle=":", alpha=0.7, label="Perfect")
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)

    def _calculate_rev(self, df, cols, ratios, clim):
        """
        Calculates REV for given ratios.
        """
        hits = df[cols[0]].sum()
        misses = df[cols[1]].sum()
        fa = df[cols[2]].sum()
        cn = df[cols[3]].sum()
        return compute_rev(hits, misses, fa, cn, ratios, clim)


# TDD Anchors:
# 1. test_rev_max_value: REV should never exceed 1.
# 2. test_rev_at_climatology: REV should be 0 if forecast equals climatology strategy.
