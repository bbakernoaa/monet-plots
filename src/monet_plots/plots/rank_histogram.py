from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from ..plot_utils import _update_history, compute, normalize_data
from ..verification_metrics import compute_rank_histogram
from .base import BasePlot


class RankHistogramPlot(BasePlot):
    """
    Rank Histogram (Talagrand Diagram).

    Visualizes the distribution of observation ranks within an ensemble.
    Supports native Xarray, Dask, and Cubed objects for lazy evaluation.

    Attributes
    ----------
    data : Union[pd.DataFrame, xr.Dataset, xr.DataArray]
        The normalized data containing ensemble ranks or raw ensemble/obs.
    """

    def __init__(
        self,
        data: Any = None,
        fig: Optional[plt.Figure] = None,
        ax: Optional[plt.Axes] = None,
        **kwargs: Any,
    ):
        """
        Initialize the RankHistogramPlot.

        Parameters
        ----------
        data : Any, optional
            Input data. Can be a pandas DataFrame, xarray Dataset,
            xarray DataArray, or numpy ndarray.
        fig : plt.Figure, optional
            Matplotlib figure to use.
        ax : plt.Axes, optional
            Matplotlib axes to use.
        **kwargs : Any
            Additional keyword arguments passed to BasePlot.

        Examples
        --------
        >>> import numpy as np
        >>> import xarray as xr
        >>> ranks = np.random.randint(0, 11, 100)
        >>> ds = xr.Dataset({"rank": (["index"], ranks)})
        >>> plot = RankHistogramPlot(ds)
        """
        super().__init__(fig=fig, ax=ax, **kwargs)
        self.data = normalize_data(data) if data is not None else None
        if self.data is not None:
            _update_history(self.data, "Initialized RankHistogramPlot")

    def plot(
        self,
        data: Optional[Any] = None,
        rank_col: str = "rank",
        ensemble_col: Optional[str] = None,
        obs_col: Optional[str] = None,
        member_dim: str = "member",
        n_members: Optional[int] = None,
        label_col: Optional[str] = None,
        normalize: bool = True,
        **kwargs: Any,
    ) -> plt.Axes:
        """
        Main plotting method for Rank Histogram (Track A).

        Parameters
        ----------
        data : Any, optional
            Data containing ranks or ensemble/obs. If None, uses the data provided at initialization.
        rank_col : str, default "rank"
            Variable name containing the pre-computed ranks.
        ensemble_col : str, optional
            Variable name containing the ensemble data.
        obs_col : str, optional
            Variable name containing the observation data.
        member_dim : str, default "member"
            Dimension name for ensemble members if using ensemble/obs.
        n_members : int, optional
            Number of ensemble members. Inferred from data if None.
        label_col : str, optional
            Variable name for grouping multiple histograms.
        normalize : bool, default True
            If True, plot relative frequency; else raw counts.
        **kwargs : Any
            Additional keyword arguments passed to `ax.bar`.

        Returns
        -------
        plt.Axes
            The Matplotlib axes containing the plot.

        Examples
        --------
        >>> plot = RankHistogramPlot(data=ds)
        >>> ax = plot.plot(rank_col="rank")
        """
        plot_data = normalize_data(data) if data is not None else self.data

        if plot_data is None:
            raise ValueError("No data provided for plotting.")

        # Vectorized Rank Computation if ensemble/obs provided
        if ensemble_col and obs_col:
            if isinstance(plot_data, xr.Dataset):
                counts = compute_rank_histogram(
                    plot_data[ensemble_col], plot_data[obs_col], member_dim=member_dim
                )
                n_members_val = plot_data[ensemble_col].sizes[member_dim]
                self._plot_bars(counts, n_members_val, normalize=normalize, **kwargs)
            else:
                raise TypeError(
                    "Vectorized rank computation requires an xarray.Dataset."
                )
        else:
            # Use pre-computed ranks
            self._plot_precomputed(
                plot_data,
                rank_col,
                n_members,
                label_col,
                normalize=normalize,
                **kwargs,
            )

        return self.ax

    def _plot_bars(
        self,
        counts: Union[np.ndarray, xr.DataArray],
        n_members: int,
        normalize: bool = True,
        label: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Internal helper to plot bars from counts."""
        num_bins = n_members + 1
        total = float(counts.sum())
        freq = counts / total if normalize else counts

        # Compute for plotting (final step)
        freq_vals = compute(freq)
        expected = 1.0 / num_bins if normalize else float(total / num_bins)

        self.ax.bar(np.arange(num_bins), freq_vals, label=label, alpha=0.7, **kwargs)

        # Expected uniform line
        self.ax.axhline(
            expected, color="k", linestyle="--", linewidth=2, label="Expected (Uniform)"
        )

        # Formatting
        self.ax.set_xlabel("Rank")
        self.ax.set_ylabel("Relative Frequency" if normalize else "Count")
        self.ax.set_xticks(np.arange(num_bins))
        self.ax.set_xlim(-0.5, n_members + 0.5)
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()

    def _plot_precomputed(
        self,
        data: Union[pd.DataFrame, xr.Dataset, xr.DataArray],
        rank_col: str,
        n_members: Optional[int],
        label_col: Optional[str],
        normalize: bool = True,
        **kwargs: Any,
    ) -> None:
        """Internal helper to plot from pre-computed ranks."""
        if isinstance(data, (xr.Dataset, xr.DataArray)):
            if n_members is None:
                n_members = int(data[rank_col].max().compute())
            num_bins = n_members + 1

            if label_col:
                groups = data.groupby(label_col)
                for label, group in groups:
                    # Optimized: use xr.DataArray.count and xr.where/xr.groupby_bins if possible,
                    # but for ranks 0..N, np.bincount on computed values is often most efficient.
                    # We only compute the specific group's ranks.
                    ranks = compute(group[rank_col])
                    counts = np.bincount(
                        np.asarray(ranks).astype(int).ravel(), minlength=num_bins
                    )
                    total = counts.sum()
                    freq = counts / total if normalize else counts
                    self.ax.bar(
                        np.arange(num_bins),
                        freq,
                        label=str(label),
                        alpha=0.7,
                        **kwargs,
                    )
            else:
                ranks = compute(data[rank_col])
                counts = np.bincount(
                    np.asarray(ranks).astype(int).ravel(), minlength=num_bins
                )
                total = counts.sum()
                freq = counts / total if normalize else counts
                self.ax.bar(np.arange(num_bins), freq, alpha=0.7, **kwargs)
        else:
            # Pandas fallback
            df = data
            if n_members is None:
                n_members = int(df[rank_col].max())
            num_bins = n_members + 1
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
            else:
                counts = (
                    df[rank_col]
                    .value_counts()
                    .reindex(np.arange(num_bins), fill_value=0)
                )
                total = counts.sum()
                freq = counts / total if normalize else counts
                self.ax.bar(counts.index, freq.values, alpha=0.7, **kwargs)

        expected = (
            1.0 / (n_members + 1) if normalize else float(len(data)) / (n_members + 1)
        )
        self.ax.axhline(
            expected, color="k", linestyle="--", linewidth=2, label="Expected (Uniform)"
        )
        self.ax.set_xlabel("Rank")
        self.ax.set_ylabel("Relative Frequency" if normalize else "Count")
        self.ax.set_xticks(np.arange(n_members + 1))
        self.ax.set_xlim(-0.5, n_members + 0.5)
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()

    def hvplot(
        self,
        rank_col: str = "rank",
        n_members: Optional[int] = None,
        label_col: Optional[str] = None,
        normalize: bool = True,
        **kwargs: Any,
    ) -> Any:
        """
        Generate an interactive Rank Histogram (Track B).

        Parameters
        ----------
        rank_col : str, default "rank"
            Variable name containing the ranks.
        n_members : int, optional
            Number of ensemble members.
        label_col : str, optional
            Variable name for grouping.
        normalize : bool, default True
            If True, plot relative frequency; else raw counts.
        **kwargs : Any
            Additional keyword arguments passed to `hvplot.bar`.

        Returns
        -------
        hv.Plot
            The HoloViews/hvPlot object.

        Examples
        --------
        >>> plot = RankHistogramPlot(data=ds)
        >>> hv_obj = plot.hvplot()
        """
        import hvplot.pandas  # noqa: F401
        import hvplot.xarray  # noqa: F401

        if self.data is None:
            raise ValueError("No data available for plotting.")

        # For Track B interaction, we often need to compute for small binned summaries
        if isinstance(self.data, (xr.Dataset, xr.DataArray)):
            if n_members is None:
                n_members = int(self.data[rank_col].max().compute())

            if label_col:
                # Grouped Xarray to DataFrame for hvplot
                # This keeps the aggregation lazy as much as possible
                df_list = []
                for label, group in self.data.groupby(label_col):
                    ranks = compute(group[rank_col])
                    counts = np.bincount(
                        np.asarray(ranks).astype(int).ravel(), minlength=n_members + 1
                    )
                    temp_df = pd.DataFrame(
                        {
                            "rank": np.arange(n_members + 1),
                            "value": counts / counts.sum() if normalize else counts,
                            label_col: str(label),
                        }
                    )
                    df_list.append(temp_df)
                plot_df = pd.concat(df_list)
            else:
                ranks = compute(self.data[rank_col])
                counts = np.bincount(
                    np.asarray(ranks).astype(int).ravel(), minlength=n_members + 1
                )
                plot_df = pd.DataFrame(
                    {
                        "rank": np.arange(n_members + 1),
                        "value": counts / counts.sum() if normalize else counts,
                    }
                )
        else:
            df = self.data
            if n_members is None:
                n_members = int(df[rank_col].max())
            num_bins = n_members + 1
            if label_col:
                plot_df_list = []
                for name, group in df.groupby(label_col):
                    counts = (
                        group[rank_col]
                        .value_counts()
                        .reindex(np.arange(num_bins), fill_value=0)
                    )
                    total = counts.sum()
                    freq = counts / total if normalize else counts
                    temp_df = freq.to_frame(name="value")
                    temp_df.index.name = "rank"
                    temp_df[label_col] = name
                    plot_df_list.append(temp_df.reset_index())
                plot_df = pd.concat(plot_df_list)
            else:
                counts = (
                    df[rank_col]
                    .value_counts()
                    .reindex(np.arange(num_bins), fill_value=0)
                )
                total = counts.sum()
                freq = counts / total if normalize else counts
                plot_df = freq.to_frame(name="value")
                plot_df.index.name = "rank"
                plot_df = plot_df.reset_index()

        return plot_df.hvplot.bar(
            x="rank",
            y="value",
            by=label_col if label_col in plot_df.columns else None,
            alpha=0.7,
            **kwargs,
        )
