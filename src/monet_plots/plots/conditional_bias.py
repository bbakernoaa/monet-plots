# src/monet_plots/plots/conditional_bias.py
"""Conditional Bias Plot for model evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ..plot_utils import normalize_data
from ..verification_metrics import _update_history, compute_binned_bias
from .base import BasePlot

if TYPE_CHECKING:
    import holoviews as hv
    import matplotlib.axes


class ConditionalBiasPlot(BasePlot):
    """
    Conditional Bias Plot.

    Visualizes the Mean Bias (Forecast - Observation) as a function of the
    Observed Value. This plot helps identify if a model has systematic
    overestimation or underestimation in specific regimes (e.g., low vs high
    values).

    Supports lazy evaluation via Dask and native Xarray objects.

    Attributes
    ----------
    data : Union[xr.DataArray, xr.Dataset, pd.DataFrame]
        The input data for the plot.
    stats : xr.Dataset
        The calculated binned statistics (mean, std, count).
    """

    def __init__(
        self,
        data: Optional[Any] = None,
        fig: Optional[plt.Figure] = None,
        ax: Optional[plt.Axes] = None,
        **kwargs: Any,
    ):
        """
        Initialize the ConditionalBiasPlot.

        Parameters
        ----------
        data : Any, optional
            Input data. Can be a pandas DataFrame, xarray DataArray,
            xarray Dataset, or numpy ndarray. Preferred format is xarray
            with dask backing for large datasets.
        fig : matplotlib.figure.Figure, optional
            An existing Figure object.
        ax : matplotlib.axes.Axes, optional
            An existing Axes object.
        **kwargs : Any
            Additional keyword arguments passed to BasePlot.
        """
        super().__init__(fig=fig, ax=ax, **kwargs)
        if self.ax is None:
            self.ax = self.fig.add_subplot(1, 1, 1)

        self.data = normalize_data(data) if data is not None else None
        self.stats: Optional[xr.Dataset] = None

        if self.data is not None:
            _update_history(self.data, "Initialized ConditionalBiasPlot")

    def _get_global_bins(self, obs_col: str, n_bins: int) -> np.ndarray:
        """Calculate global bin edges for the entire dataset."""
        obs = self.data[obs_col]

        # Check for dask safely
        is_dask = False
        if isinstance(obs, xr.DataArray) and hasattr(obs.data, "chunks"):
            is_dask = True

        if is_dask:
            import dask

            o_min, o_max = dask.compute(obs.min(), obs.max())
        else:
            o_min, o_max = obs.min(), obs.max()

        # Ensure they are scalar floats
        o_min, o_max = float(o_min), float(o_max)

        if o_min == o_max:
            return np.array([o_min - 0.5, o_min + 0.5])

        return np.linspace(o_min, o_max, n_bins + 1)

    def _validate_cols(self, *cols: str) -> None:
        """Check if required columns/variables exist in data."""
        available = []
        if isinstance(self.data, (xr.Dataset, xr.DataArray)):
            if isinstance(self.data, xr.Dataset):
                available = list(self.data.data_vars) + list(self.data.coords)
            else:
                available = (
                    [self.data.name] + list(self.data.coords)
                    if self.data.name
                    else list(self.data.coords)
                )
        elif hasattr(self.data, "columns"):
            available = list(self.data.columns)

        missing = [c for c in cols if c not in available]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def plot(
        self,
        data: Optional[Any] = None,
        obs_col: str = "obs",
        fcst_col: str = "fcst",
        n_bins: int = 10,
        label_col: Optional[str] = None,
        min_samples: int = 5,
        **kwargs: Any,
    ) -> matplotlib.axes.Axes:
        """
        Generate the conditional bias plot using Matplotlib.

        This method calculates binned statistics and renders them as an
        errorbar plot. For Dask-backed data, computations are deferred
        until the final plotting call.

        Parameters
        ----------
        data : Any, optional
            Input data (if not provided during initialization).
        obs_col : str, optional
            Variable or column name for observations, by default 'obs'.
        fcst_col : str, optional
            Variable or column name for forecasts, by default 'fcst'.
        n_bins : int, optional
            Number of bins for observed values, by default 10.
        label_col : str, optional
            Grouping variable/column for multiple curves (e.g., 'model_id').
        min_samples : int, optional
            Minimum number of samples required in a bin to plot it, by default 5.
        **kwargs : Any
            Additional keyword arguments passed to `ax.errorbar`.

        Returns
        -------
        matplotlib.axes.Axes
            The axes object with the conditional bias plot.
        """
        if data is not None:
            self.data = normalize_data(data)

        if self.data is None:
            raise ValueError("Data must be provided either in __init__ or plot()")

        if len(self.data) == 0:
            raise ValueError("Data cannot be empty")

        self._validate_cols(obs_col, fcst_col)
        if label_col:
            self._validate_cols(label_col)

        # 1. Calculate Binned Statistics
        bins = self._get_global_bins(obs_col, n_bins)

        if label_col:
            if isinstance(self.data, (xr.DataArray, xr.Dataset)):
                results = []
                ds = (
                    self.data
                    if isinstance(self.data, xr.Dataset)
                    else self.data.to_dataset(name="data")
                )

                for val, group in ds.groupby(label_col):
                    stats = compute_binned_bias(
                        group[obs_col], group[fcst_col], bins=bins
                    )
                    stats = stats.assign_coords({label_col: val}).expand_dims(label_col)
                    results.append(stats)
                self.stats = xr.concat(results, dim=label_col)
            else:
                df = self.data
                results = []
                for name, group in df.groupby(label_col):
                    obs = xr.DataArray(group[obs_col].values, dims="sample")
                    mod = xr.DataArray(group[fcst_col].values, dims="sample")
                    stats = compute_binned_bias(obs, mod, bins=bins)
                    stats = stats.assign_coords({label_col: name}).expand_dims(
                        label_col
                    )
                    results.append(stats)
                self.stats = xr.concat(results, dim=label_col)
        else:
            obs = self.data[obs_col]
            mod = self.data[fcst_col]
            if not isinstance(obs, xr.DataArray):
                obs = xr.DataArray(obs, dims="sample")
            if not isinstance(mod, xr.DataArray):
                mod = xr.DataArray(mod, dims="sample")
            self.stats = compute_binned_bias(obs, mod, bins=bins)

        # 2. Rendering
        if label_col:
            for i in range(len(self.stats[label_col])):
                sub = self.stats.isel({label_col: i})
                label_val = sub[label_col].values
                label_str = (
                    str(label_val.item())
                    if hasattr(label_val, "item")
                    else str(label_val)
                )
                self._draw_errorbar(sub, label_str, min_samples, **kwargs)
        else:
            self._draw_errorbar(self.stats, "Model", min_samples, **kwargs)

        # 3. Finalize UI
        self.ax.axhline(
            0, color="k", linestyle="--", linewidth=1.5, alpha=0.8, label="No Bias"
        )
        self.ax.set_xlabel(f"Observed Value ({obs_col})")
        self.ax.set_ylabel("Mean Bias (Forecast - Observation)")
        self.ax.grid(True, linestyle=":", alpha=0.6)
        self.ax.legend(loc="best")

        _update_history(self.stats, "Generated Matplotlib ConditionalBiasPlot")
        return self.ax

    def _draw_errorbar(
        self, stats: xr.Dataset, label: str, min_samples: int, **kwargs: Any
    ) -> None:
        """Helper to draw a single errorbar curve."""
        # Compute everything first to avoid lazy indexing issues
        stats_comp = stats.compute()
        mask = stats_comp["count"] >= min_samples
        valid_stats = stats_comp.where(mask, drop=True)

        if len(valid_stats.bin_center) > 0:
            self.ax.errorbar(
                valid_stats.bin_center,
                valid_stats["mean"],
                yerr=valid_stats["std"],
                fmt="o-",
                capsize=4,
                label=label,
                **kwargs,
            )

    def hvplot(
        self,
        obs_col: str = "obs",
        fcst_col: str = "fcst",
        n_bins: int = 10,
        label_col: Optional[str] = None,
        min_samples: int = 5,
        **kwargs: Any,
    ) -> hv.Element:
        """
        Generate an interactive conditional bias plot using hvPlot.

        This method follows Track B of the Aero Protocol, providing an
        interactive visualization suitable for exploration in notebooks.

        Parameters
        ----------
        obs_col : str, optional
            Variable name for observations, by default 'obs'.
        fcst_col : str, optional
            Variable name for forecasts, by default 'fcst'.
        n_bins : int, optional
            Number of bins for observed values, by default 10.
        label_col : str, optional
            Grouping variable for multiple curves.
        min_samples : int, optional
            Minimum samples per bin, by default 5.
        **kwargs : Any
            Additional keyword arguments passed to `hvplot.errorbars`.

        Returns
        -------
        holoviews.Element
            The interactive HoloViews object.
        """
        import hvplot.xarray  # noqa: F401

        if self.data is None:
            raise ValueError("Data must be provided during initialization for hvplot()")

        bins = self._get_global_bins(obs_col, n_bins)

        if label_col:
            results = []
            ds = (
                self.data
                if isinstance(self.data, xr.Dataset)
                else self.data.to_dataset(name="data")
            )
            for val, group in ds.groupby(label_col):
                stats = compute_binned_bias(group[obs_col], group[fcst_col], bins=bins)
                stats = stats.assign_coords({label_col: val}).expand_dims(label_col)
                results.append(stats)
            plot_stats = xr.concat(results, dim=label_col)
        else:
            plot_stats = compute_binned_bias(
                self.data[obs_col], self.data[fcst_col], bins=bins
            )

        # Filtering in hvplot can be done lazily if needed, but we often want
        # to filter before passing to hvplot to keep the UI clean.
        plot_stats = plot_stats.where(plot_stats["count"] >= min_samples, drop=True)

        main_plot = plot_stats.hvplot.errorbars(
            x="bin_center",
            y="mean",
            yerr1="std",
            by=label_col if label_col else None,
            xlabel=f"Observed Value ({obs_col})",
            ylabel="Mean Bias (Forecast - Observation)",
            title="Conditional Bias",
            **kwargs,
        )

        import holoviews as hv

        hline = hv.HLine(0).opts(color="black", line_dash="dashed")

        return hline * main_plot
