import numpy as np
import pandas as pd
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

    def __init__(
        self,
        data: Any,
        counts_cols: List[str] = ["hits", "misses", "fa", "cn"],
        climatology: Optional[float] = None,
        label_col: Optional[str] = None,
        cost_loss_ratios: Optional[np.ndarray] = None,
        fig=None,
        ax=None,
        **kwargs,
    ):
        super().__init__(fig=fig, ax=ax, **kwargs)
        self.data = to_dataframe(data)
        self.counts_cols = counts_cols
        self.climatology = climatology
        self.label_col = label_col
        self.cost_loss_ratios = cost_loss_ratios

    def plot(self, **kwargs):
        """
        Main plotting method.

        Args:
            **kwargs: Matplotlib kwargs.
        """
        validate_dataframe(self.data, required_columns=self.counts_cols)

        if self.climatology is None:
            total_events = (
                self.data[self.counts_cols[0]].sum()
                + self.data[self.counts_cols[1]].sum()
            )
            total = (
                total_events
                + self.data[self.counts_cols[2]].sum()
                + self.data[self.counts_cols[3]].sum()
            )
            climatology = total_events / total if total > 0 else 0.5
        else:
            climatology = self.climatology

        if self.cost_loss_ratios is None:
            cost_loss_ratios = np.linspace(0.001, 0.999, 100)
        else:
            cost_loss_ratios = self.cost_loss_ratios

        # TDD Anchor: Test REV calculation logic

        if self.label_col:
            for name, group in self.data.groupby(self.label_col):
                rev_values = self._calculate_rev(
                    group, self.counts_cols, cost_loss_ratios, climatology
                )
                self.ax.plot(cost_loss_ratios, rev_values, label=str(name), **kwargs)
            self.ax.legend(loc="best")
        else:
            rev_values = self._calculate_rev(
                self.data, self.counts_cols, cost_loss_ratios, climatology
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

    def hvplot(self, **kwargs):
        """Generate an interactive Relative Economic Value plot using hvPlot."""
        import hvplot.pandas  # noqa: F401
        import holoviews as hv

        if self.climatology is None:
            total_events = (
                self.data[self.counts_cols[0]].sum()
                + self.data[self.counts_cols[1]].sum()
            )
            total = (
                total_events
                + self.data[self.counts_cols[2]].sum()
                + self.data[self.counts_cols[3]].sum()
            )
            climatology = total_events / total if total > 0 else 0.5
        else:
            climatology = self.climatology

        if self.cost_loss_ratios is None:
            cost_loss_ratios = np.linspace(0.001, 0.999, 100)
        else:
            cost_loss_ratios = self.cost_loss_ratios

        all_revs = []
        if self.label_col:
            for name, group in self.data.groupby(self.label_col):
                rev_values = self._calculate_rev(
                    group, self.counts_cols, cost_loss_ratios, climatology
                )
                temp_df = pd.DataFrame(
                    {"Cost/Loss Ratio": cost_loss_ratios, "REV": rev_values}
                )
                temp_df[self.label_col] = str(name)
                all_revs.append(temp_df)
            df_plot = pd.concat(all_revs)
        else:
            rev_values = self._calculate_rev(
                self.data, self.counts_cols, cost_loss_ratios, climatology
            )
            df_plot = pd.DataFrame(
                {"Cost/Loss Ratio": cost_loss_ratios, "REV": rev_values}
            )

        plot_kwargs = {
            "x": "Cost/Loss Ratio",
            "y": "REV",
            "kind": "line",
            "title": "Relative Economic Value (REV)",
            "ylim": (-0.2, 1.05),
        }
        if self.label_col:
            plot_kwargs["by"] = self.label_col

        plot_kwargs.update(kwargs)

        p = df_plot.hvplot(**plot_kwargs)
        clim_line = hv.HLine(0).opts(color="black", alpha=0.5, line_dash="dashed")
        perfect_line = hv.HLine(1).opts(color="gray", alpha=0.5, line_dash="dotted")

        return p * clim_line * perfect_line
