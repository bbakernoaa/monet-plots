import numpy as np
import pandas as pd
from typing import Optional, Any
from .base import BasePlot
from ..plot_utils import validate_dataframe, to_dataframe


class RankHistogramPlot(BasePlot):
    """
    Rank Histogram (Talagrand Diagram).

    Visualizes the distribution of observation ranks within an ensemble.

    Functional Requirements:
    1. Plot bar chart of rank frequencies.
    2. Draw horizontal line for "Perfect Flatness" (uniform distribution).
    3. Support normalizing frequencies (relative frequency) or raw counts.
    4. Interpret shapes: U-shape (underdispersed), A-shape (overdispersed), Bias (slope).

    Edge Cases:
    - Unequal ensemble sizes (requires binning or normalization logic, but typically preprocessing handles this).
    - Missing ranks (should be 0 height bars).
    """

    def __init__(
        self,
        data: Any,
        rank_col: str = "rank",
        n_members: Optional[int] = None,
        label_col: Optional[str] = None,
        normalize: bool = True,
        fig=None,
        ax=None,
        **kwargs,
    ):
        super().__init__(fig=fig, ax=ax, **kwargs)
        self.data = to_dataframe(data)
        self.rank_col = rank_col
        self.n_members = n_members
        self.label_col = label_col
        self.normalize = normalize
        validate_dataframe(self.data, required_columns=[self.rank_col])

    def plot(self, **kwargs):
        """
        Main plotting method.

        Args:
            **kwargs: Matplotlib kwargs.
        """
        if self.n_members is None:
            n_members = int(self.data[self.rank_col].max())
        else:
            n_members = self.n_members

        num_bins = n_members + 1

        if self.normalize:
            expected = 1.0 / num_bins
        else:
            expected = len(self.data) / num_bins

        # TDD Anchor: Validate inputs

        if self.label_col:
            for name, group in self.data.groupby(self.label_col):
                counts = (
                    group[self.rank_col]
                    .value_counts()
                    .reindex(np.arange(num_bins), fill_value=0)
                )
                total = counts.sum()
                freq = counts / total if self.normalize else counts
                self.ax.bar(
                    counts.index, freq.values, label=str(name), alpha=0.7, **kwargs
                )
            self.ax.legend()
        else:
            counts = (
                self.data[self.rank_col]
                .value_counts()
                .reindex(np.arange(num_bins), fill_value=0)
            )
            total = counts.sum()
            freq = counts / total if self.normalize else counts
            self.ax.bar(counts.index, freq.values, alpha=0.7, **kwargs)

        # Expected uniform line
        self.ax.axhline(
            expected, color="k", linestyle="--", linewidth=2, label="Expected (Uniform)"
        )
        self.ax.legend()

        # Formatting
        self.ax.set_xlabel("Rank")
        self.ax.set_ylabel("Relative Frequency" if self.normalize else "Count")
        self.ax.set_xticks(np.arange(n_members + 1))
        self.ax.set_xlim(-0.5, n_members + 0.5)
        self.ax.grid(True, alpha=0.3)

        # TDD Anchor: test_normalization: Check sum of frequencies is 1.0.
        # TDD Anchor: test_missing_ranks: Ensure ranks with 0 counts are plotted as 0.

    # TDD Anchors:
    # 1. test_flat_distribution: Verify perfectly uniform ranks yield flat line.
    # 2. test_normalization: Check sum of frequencies is 1.0.
    # 3. test_missing_ranks: Ensure ranks with 0 counts are plotted as 0.

    def hvplot(self, **kwargs):
        """Generate an interactive rank histogram using hvPlot."""
        import hvplot.pandas  # noqa: F401
        import holoviews as hv

        if self.n_members is None:
            n_members = int(self.data[self.rank_col].max())
        else:
            n_members = self.n_members
        num_bins = n_members + 1

        plot_dfs = []
        if self.label_col:
            for name, group in self.data.groupby(self.label_col):
                counts = (
                    group[self.rank_col]
                    .value_counts()
                    .reindex(np.arange(num_bins), fill_value=0)
                )
                total = counts.sum()
                freq = counts / total if self.normalize else counts
                temp_df = freq.reset_index()
                temp_df.columns = ["rank", "value"]
                temp_df[self.label_col] = name
                plot_dfs.append(temp_df)
            final_df = pd.concat(plot_dfs)
        else:
            counts = (
                self.data[self.rank_col]
                .value_counts()
                .reindex(np.arange(num_bins), fill_value=0)
            )
            total = counts.sum()
            freq = counts / total if self.normalize else counts
            final_df = freq.reset_index()
            final_df.columns = ["rank", "value"]

        plot_kwargs = {
            "x": "rank",
            "y": "value",
            "kind": "bar",
            "xlabel": "Rank",
            "ylabel": "Relative Frequency" if self.normalize else "Count",
            "title": "Rank Histogram",
        }
        if self.label_col:
            plot_kwargs["by"] = self.label_col

        plot_kwargs.update(kwargs)

        p = final_df.hvplot(**plot_kwargs)

        expected_val = 1.0 / num_bins if self.normalize else len(self.data) / num_bins
        expected_line = hv.HLine(expected_val).opts(
            color="black", alpha=0.5, line_dash="dashed"
        )

        return p * expected_line
