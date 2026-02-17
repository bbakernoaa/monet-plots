import numpy as np
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

    def __init__(self, fig=None, ax=None, **kwargs):
        super().__init__(fig=fig, ax=ax, **kwargs)

    def plot(
        self,
        data: Any,
        rank_col: str = "rank",
        n_members: Optional[int] = None,
        label_col: Optional[str] = None,
        normalize: bool = True,
        **kwargs,
    ):
        """
        Main plotting method.

        Args:
            data (pd.DataFrame, np.ndarray, xr.Dataset, xr.DataArray): Data containing ranks (0 to n_members).
            rank_col (str): Column containing the rank of the observation.
            n_members (Optional[int]): Number of ensemble members (defines n_bins = n_members + 1).
                                      Inferred from max(rank) if None.
            label_col (Optional[str]): Grouping for multiple histograms (e.g., lead times).
            normalize (bool): If True, plot relative frequency; else raw counts.
            **kwargs: Matplotlib kwargs.
        """
        df = to_dataframe(data)
        validate_dataframe(df, required_columns=[rank_col])

        if n_members is None:
            n_members = int(df[rank_col].max())

        num_bins = n_members + 1

        if normalize:
            expected = 1.0 / num_bins
        else:
            expected = len(df) / num_bins

        # TDD Anchor: Validate inputs

        if label_col:
            for name, group in df.groupby(label_col):
                counts = (
                    group[rank_col]
                    .value_counts()
                    .reindex(np.arange(num_bins), fill_value=0)
                )
                total = counts.sum()
                freq = counts / total if normalize else counts
                self.ax.bar(
                    counts.index, freq.values, label=str(name), alpha=0.7, **kwargs
                )
            self.ax.legend()
        else:
            counts = (
                df[rank_col].value_counts().reindex(np.arange(num_bins), fill_value=0)
            )
            total = counts.sum()
            freq = counts / total if normalize else counts
            self.ax.bar(counts.index, freq.values, alpha=0.7, **kwargs)

        # Expected uniform line
        self.ax.axhline(
            expected, color="k", linestyle="--", linewidth=2, label="Expected (Uniform)"
        )
        self.ax.legend()

        # Formatting
        self.ax.set_xlabel("Rank")
        self.ax.set_ylabel("Relative Frequency" if normalize else "Count")
        self.ax.set_xticks(np.arange(n_members + 1))
        self.ax.set_xlim(-0.5, n_members + 0.5)
        self.ax.grid(True, alpha=0.3)

        # TDD Anchor: test_normalization: Check sum of frequencies is 1.0.
        # TDD Anchor: test_missing_ranks: Ensure ranks with 0 counts are plotted as 0.


# TDD Anchors:
# 1. test_flat_distribution: Verify perfectly uniform ranks yield flat line.
# 2. test_normalization: Check sum of frequencies is 1.0.
# 3. test_missing_ranks: Ensure ranks with 0 counts are plotted as 0.
