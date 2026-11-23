from math import ceil
from typing import Any, Optional

import numpy as np
from matplotlib.ticker import PercentFormatter

from ..plot_utils import to_dataframe, validate_dataframe
from .base import BasePlot, figure_style


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
        self._owns_axes = fig is None and ax is None

    @figure_style
    def plot(
        self,
        data: Any,
        rank_col: str = "rank",
        n_members: Optional[int] = None,
        label_col: Optional[str] = None,
        normalize: bool = True,
        *,
        facet_col: str | None = None,
        col_wrap: int = 2,
        title: str | None = None,
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
            facet_col: Draw one panel per group on common rank and frequency scales.
            col_wrap: Maximum number of panel columns.
            title: Figure title for facets, or axes title for a single panel.
            **kwargs: Matplotlib kwargs.
        """
        df = to_dataframe(data)
        validate_dataframe(df, required_columns=[rank_col])

        if n_members is None:
            n_members = int(df[rank_col].max())

        if facet_col is not None:
            if label_col is not None:
                raise ValueError("Choose either facet_col or label_col.")
            validate_dataframe(df, required_columns=[facet_col, rank_col])
            self._plot_facets(
                df, rank_col, n_members, normalize, facet_col, col_wrap, title, kwargs
            )
            return

        num_bins = n_members + 1

        if normalize:
            expected = 1.0 / num_bins
        else:
            expected = df[rank_col].isin(range(num_bins)).sum() / num_bins

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
        if self._figure_style is not None:
            self.ax.grid(False, axis="x")
            self.ax.set_axisbelow(True)
            if normalize:
                self.ax.yaxis.set_major_formatter(PercentFormatter(1))
        if title is not None:
            self.ax.set_title(title, loc="left")

        # TDD Anchor: test_normalization: Check sum of frequencies is 1.0.
        # TDD Anchor: test_missing_ranks: Ensure ranks with 0 counts are plotted as 0.

    def _plot_facets(
        self, df, rank_col, n_members, normalize, facet_col, col_wrap, title, options
    ):
        if not self._owns_axes:
            raise ValueError("Facets require a RankHistogramPlot with its own figure.")
        groups = list(df.groupby(facet_col, observed=True, sort=False))
        if not groups or col_wrap < 1:
            raise ValueError(
                "Facets require at least one group and a positive col_wrap."
            )
        ncols = min(col_wrap, len(groups))
        nrows = ceil(len(groups) / ncols)
        self.fig.clear()
        self.fig.set_layout_engine("constrained")
        axes = self.fig.subplots(nrows, ncols, sharex=True, sharey=True, squeeze=False)
        for ax, (name, group) in zip(axes.flat, groups):
            self.ax = ax
            sample_count = group[rank_col].isin(range(n_members + 1)).sum()
            self.plot(
                group,
                rank_col=rank_col,
                n_members=n_members,
                normalize=normalize,
                title=f"{name} · n={sample_count:,}",
                **options,
            )
            ax.get_legend().remove()
            ax.label_outer()
        for ax in list(axes.flat)[len(groups) :]:
            ax.set_visible(False)
        self.ax = axes.flat[0]
        heading = title or f"Ensemble rank histograms · {n_members} members"
        self.fig.suptitle(heading + "\nDashed line: expected uniform ranks")


# TDD Anchors:
# 1. test_flat_distribution: Verify perfectly uniform ranks yield flat line.
# 2. test_normalization: Check sum of frequencies is 1.0.
# 3. test_missing_ranks: Ensure ranks with 0 counts are plotted as 0.
