import pandas as pd
from typing import Optional, Any
from .base import BasePlot
from ..plot_utils import validate_dataframe, to_dataframe


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

    def __init__(
        self,
        data: Any,
        obs_col: str,
        fcst_col: str,
        n_bins: int = 10,
        label_col: Optional[str] = None,
        fig=None,
        ax=None,
        **kwargs,
    ):
        super().__init__(fig=fig, ax=ax, **kwargs)
        self.data = to_dataframe(data)
        self.obs_col = obs_col
        self.fcst_col = fcst_col
        self.n_bins = n_bins
        self.label_col = label_col
        validate_dataframe(self.data, required_columns=[self.obs_col, self.fcst_col])

    def plot(self, **kwargs):
        """
        Main plotting method.

        Args:
            **kwargs: Matplotlib kwargs.
        """
        df_plot = self.data.copy()
        df_plot["bias"] = df_plot[self.fcst_col] - df_plot[self.obs_col]

        if self.label_col:
            for name, group in df_plot.groupby(self.label_col):
                self._plot_binned_bias(
                    group, self.obs_col, "bias", self.n_bins, label=str(name), **kwargs
                )
            self.ax.legend(loc="best")
        else:
            self._plot_binned_bias(
                df_plot, self.obs_col, "bias", self.n_bins, label="Model", **kwargs
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

    def hvplot(self, **kwargs):
        """Generate an interactive conditional bias plot using hvPlot."""
        import hvplot.pandas  # noqa: F401
        import holoviews as hv

        df_plot = self.data.copy()
        df_plot["bias"] = df_plot[self.fcst_col] - df_plot[self.obs_col]

        all_binned = []
        if self.label_col:
            for name, group in df_plot.groupby(self.label_col):
                bins = pd.cut(group[self.obs_col], bins=self.n_bins, duplicates="drop")
                binned = (
                    group.groupby(bins, observed=False)["bias"]
                    .agg(["mean", "std", "count"])
                    .reset_index()
                )
                binned["bin_center"] = binned[self.obs_col].apply(
                    lambda interval: interval.mid
                )
                binned = binned[binned["count"] >= 5]
                binned[self.label_col] = str(name)
                all_binned.append(binned)
            final_df = pd.concat(all_binned)
        else:
            bins = pd.cut(df_plot[self.obs_col], bins=self.n_bins, duplicates="drop")
            final_df = (
                df_plot.groupby(bins, observed=False)["bias"]
                .agg(["mean", "std", "count"])
                .reset_index()
            )
            final_df["bin_center"] = final_df[self.obs_col].apply(
                lambda interval: interval.mid
            )
            final_df = final_df[final_df["count"] >= 5]

        plot_kwargs = {
            "x": "bin_center",
            "y": "mean",
            "kind": "scatter",
            "xlabel": "Observed Value",
            "ylabel": "Mean Bias (Forecast - Observation)",
            "title": "Conditional Bias",
        }
        if self.label_col:
            plot_kwargs["by"] = self.label_col

        plot_kwargs.update(kwargs)

        p = final_df.hvplot(**plot_kwargs)
        zero_line = hv.HLine(0).opts(color="black", alpha=0.5, line_dash="dashed")

        return p * zero_line
