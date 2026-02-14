from typing import Any, Optional

import pandas as pd

from ..plot_utils import to_dataframe, validate_dataframe
from .base import BasePlot


class ConditionalBiasPlot(BasePlot):
    """
    Conditional Bias Plot.

    Visualizes the Bias (Forecast - Observation) as a function of the Observed Value.
    Also known as a Conditional Quantile Plot or Bias-Variance decomposition plot in some contexts.

    Functional Requirements:
    1. Plot Bias (y-axis) vs Observed Value (x-axis).
    2. Usually binned: x-axis bins of observed values, y-axis mean bias in that bin.
    3. Include error bars (std dev or CI) for bias in each bin.
    4. Draw zero-bias line.

    Edge Cases:
    - Empty bins (no observations in range).
    - Outliers skewing mean bias.
    """

    def __init__(self, fig=None, ax=None, **kwargs):
        super().__init__(fig=fig, ax=ax, **kwargs)

    def plot(
        self,
        data: Any,
        obs_col: str,
        fcst_col: str,
        n_bins: int = 10,
        label_col: Optional[str] = None,
        **kwargs,
    ):
        """
        Main plotting method.

        Args:
            data (pd.DataFrame, np.ndarray, xr.Dataset, xr.DataArray): Paired obs/fcst data.
            obs_col (str): Observation column.
            fcst_col (str): Forecast column.
            n_bins (int): Number of bins for observed values.
            label_col (str): Grouping column.
            **kwargs: Matplotlib kwargs.
        """
        df = to_dataframe(data)
        validate_dataframe(df, required_columns=[obs_col, fcst_col])

        df_plot = df.copy()
        df_plot["bias"] = df_plot[fcst_col] - df_plot[obs_col]

        if label_col:
            for name, group in df_plot.groupby(label_col):
                self._plot_binned_bias(
                    group, obs_col, "bias", n_bins, label=str(name), **kwargs
                )
            self.ax.legend(loc="best")
        else:
            self._plot_binned_bias(
                df_plot, obs_col, "bias", n_bins, label="Model", **kwargs
            )

        self.ax.axhline(
            0, color="k", linestyle="--", linewidth=2, alpha=0.8, label="No Bias"
        )
        self.ax.legend()
        self.ax.set_xlabel("Observed Value")
        self.ax.set_ylabel("Mean Bias (Forecast - Observation)")
        self.ax.grid(True, alpha=0.3)

    def _plot_binned_bias(self, df, x_col, y_col, n_bins, label, **kwargs):
        """
        Helper to bin data and plot mean +/- std.
        """
        # Create bins
        bins = pd.cut(df[x_col], bins=n_bins, duplicates="drop")
        binned = (
            df.groupby(bins, observed=False)[y_col]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        binned["bin_center"] = binned[x_col].apply(lambda interval: interval.mid)

        # Filter bins with sufficient samples
        binned = binned[binned["count"] >= 5]  # Arbitrary threshold

        if len(binned) > 0:
            self.ax.errorbar(
                binned["bin_center"],
                binned["mean"],
                yerr=binned["std"],
                fmt="o-",
                capsize=5,
                label=label,
                **kwargs,
            )


# TDD Anchors:
# 1. test_zero_bias_line: Verify line exists at y=0.
# 2. test_binning_consistency: Ensure bins cover full range of obs.
