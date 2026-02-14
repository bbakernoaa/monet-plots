from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from ..plot_utils import to_dataframe, validate_dataframe
from .base import BasePlot

if TYPE_CHECKING:
    import matplotlib.axes


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
        **kwargs: Any,
    ) -> matplotlib.axes.Axes:
        """
        Main plotting method.

        Parameters
        ----------
        data : Any
            Input data. Can be a pandas DataFrame, xarray DataArray,
            xarray Dataset, or numpy ndarray.
        rank_col : str, optional
            Column name for ranks (0 to n_members), by default "rank".
        n_members : int, optional
            Number of ensemble members (defines n_bins = n_members + 1).
            Inferred from max(rank) if None.
        label_col : str, optional
            Grouping for multiple histograms (e.g., lead times).
        normalize : bool, optional
            If True, plot relative frequency; else raw counts, by default True.
        **kwargs : Any
            Additional keyword arguments passed to ax.bar.

        Returns
        -------
        matplotlib.axes.Axes
            The axes object with the rank histogram.
        """
        import xarray as xr

        # Track provenance if Xarray
        if isinstance(data, (xr.DataArray, xr.Dataset)):
            history = data.attrs.get("history", "")
            data.attrs["history"] = f"Generated RankHistogramPlot; {history}"

        if isinstance(data, (xr.DataArray, xr.Dataset)):
            # Native Xarray path - maintain laziness
            ds = data
            if isinstance(ds, xr.Dataset):
                ranks = ds[rank_col]
            else:
                ranks = ds

            if n_members is None:
                n_members = int(ranks.max().compute())

            num_bins = n_members + 1
            bins = np.arange(num_bins + 1) - 0.5

            if label_col:
                labels_concrete = ds[label_col].values
                unique_labels = np.unique(labels_concrete)
                for label in unique_labels:
                    mask = labels_concrete == label
                    r_sub = ranks.isel({ranks.dims[0]: mask})
                    if hasattr(r_sub.data, "chunks"):
                        import dask.array as da

                        counts, _ = da.histogram(r_sub.data.ravel(), bins=bins)
                        counts = counts.compute()
                    else:
                        counts, _ = np.histogram(r_sub.values.ravel(), bins=bins)

                    total = counts.sum()
                    freq = counts / total if normalize else counts
                    self.ax.bar(
                        np.arange(num_bins),
                        freq,
                        label=str(label),
                        alpha=0.7,
                        **kwargs,
                    )
                self.ax.legend()
                # Use mean expected for simplicity if normalizing, otherwise hard to define
                # for multiple groups with different sizes on one axis without normalization.
                expected = (
                    1.0 / num_bins
                    if normalize
                    else ranks.size / num_bins / len(unique_labels)
                )
            else:
                if hasattr(ranks.data, "chunks"):
                    import dask.array as da

                    counts, _ = da.histogram(ranks.data.ravel(), bins=bins)
                    counts = counts.compute()
                else:
                    counts, _ = np.histogram(ranks.values.ravel(), bins=bins)

                total = counts.sum()
                freq = counts / total if normalize else counts
                self.ax.bar(np.arange(num_bins), freq, alpha=0.7, **kwargs)
                expected = 1.0 / num_bins if normalize else total / num_bins
        else:
            # Fallback path
            df = to_dataframe(data)
            validate_dataframe(df, required_columns=[rank_col])

            if n_members is None:
                n_members = int(df[rank_col].max())

            num_bins = n_members + 1

            if label_col:
                unique_labels = df[label_col].unique()
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
                expected = (
                    1.0 / num_bins
                    if normalize
                    else len(df) / num_bins / len(unique_labels)
                )
            else:
                counts = (
                    df[rank_col]
                    .value_counts()
                    .reindex(np.arange(num_bins), fill_value=0)
                )
                total = counts.sum()
                freq = counts / total if normalize else counts
                self.ax.bar(counts.index, freq.values, alpha=0.7, **kwargs)
                expected = 1.0 / num_bins if normalize else len(df) / num_bins

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
