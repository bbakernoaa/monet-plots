from __future__ import annotations

from typing import Any, Optional

import pandas as pd
import xarray as xr

from ..plot_utils import normalize_data
from ..verification_metrics import compute_binned_bias
from .base import BasePlot


class ConditionalBiasPlot(BasePlot):
    """
    Conditional Bias Plot (Unified API).

    Visualizes the bias (var2 - var1) as a function of var1 (e.g., forecast - observation).
    Supports native Xarray/Dask objects and interactive visualization.

    Parameters
    ----------
    data : Any
        The input data (Dataset, DataArray, DataFrame, or ndarray).
    var1 : str
        Name of the first variable (e.g., observations).
    var2 : str
        Name of the second variable (e.g., model/forecast).
    fig : matplotlib.figure.Figure, optional
        Figure to plot on.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    df : Any, optional
        Deprecated alias for ``data``.
    **kwargs : Any
        Additional keyword arguments for the figure.
    """

    def __init__(
        self,
        data: Optional[Any] = None,
        var1: Optional[str] = None,
        var2: Optional[str] = None,
        fig=None,
        ax=None,
        df: Any = None,
        obs_col: Optional[str] = None,  # legacy alias
        fcst_col: Optional[str] = None,  # legacy alias
        **kwargs,
    ):
        super().__init__(fig=fig, ax=ax, **kwargs)
        if df is not None and data is None:
            data = df
        self.data = normalize_data(data) if data is not None else None
        # Support legacy aliases
        self.var1 = var1 or obs_col
        self.var2 = var2 or fcst_col

    def plot(
        self,
        data: Optional[Any] = None,
        var1: Optional[str] = None,
        var2: Optional[str] = None,
        n_bins: int = 10,
        label: str = "Model",
        label_col: Optional[str] = None,
        **kwargs,
    ):
        """
        Generates the static Matplotlib plot.

        Parameters
        ----------
        data : Any, optional
            Input data, overrides self.data if provided.
        obs_col : str, optional
            Name of the observation variable. Required for Dataset/DataFrame.
        fcst_col : str, optional
            Name of the forecast variable. Required for Dataset/DataFrame.
        n_bins : int, optional
            Number of bins for observed values, by default 10.
        label : str, optional
            Label for the model data, by default "Model".
        label_col : str, optional
            Column name to group by for plotting multiple lines.
        **kwargs : Any
            Additional Matplotlib plotting arguments passed to `errorbar`.

        Returns
        -------
        matplotlib.axes.Axes
            The axes object containing the plot.
        """

        plot_data = normalize_data(data) if data is not None else self.data
        if plot_data is None:
            raise ValueError("No data provided.")

        # Allow override of var1/var2 at plot time
        var1 = var1 or self.var1
        var2 = var2 or self.var2
        if var1 is None or var2 is None:
            raise ValueError("Both 'var1' and 'var2' must be specified.")

        try:
            if label_col:
                # Handle grouping for multiple models/categories
                for name, group in plot_data.groupby(label_col):
                    v1 = group[var1]
                    v2 = group[var2]
                    self._plot_single(v1, v2, n_bins, label=str(name), **kwargs)
            else:
                # Single model plot
                if isinstance(plot_data, xr.Dataset):
                    v1 = plot_data[var1]
                    v2 = plot_data[var2]
                elif isinstance(plot_data, xr.DataArray):
                    v2 = plot_data
                    v1 = kwargs.pop("var1_data", None)
                    if v1 is None:
                        raise ValueError(
                            "var1_data must be provided if data is a DataArray."
                        )
                elif isinstance(plot_data, pd.DataFrame):
                    v1 = plot_data[var1]
                    v2 = plot_data[var2]
                else:
                    raise TypeError(f"Unsupported data type: {type(plot_data)}")

                self._plot_single(v1, v2, n_bins, label=label, **kwargs)
        except KeyError as e:
            raise ValueError(f"Required column not found: {e}") from e

        self.ax.axhline(0, color="k", linestyle="--", linewidth=1.5, alpha=0.7)

        xlabel = var1
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(f"Mean Bias ({var2} - {var1})")

        handles, labels = self.ax.get_legend_handles_labels()
        if handles:
            self.ax.legend()

        return self.ax

    def _plot_single(self, obs, mod, n_bins, label, **kwargs):
        """Helper to plot a single binned bias line."""
        stats = compute_binned_bias(obs, mod, n_bins=n_bins)
        pdf = stats.compute().dropna(dim="bin_center")

        # Filter for count > 1 to avoid showing bins with only one sample (no std dev)
        pdf = pdf.where(pdf.bias_count > 1, drop=True)

        if pdf.bin_center.size > 0:
            self.ax.errorbar(
                pdf.bin_center,
                pdf.bias_mean,
                yerr=pdf.bias_std,
                fmt="o-",
                capsize=5,
                label=label,
                **kwargs,
            )

    def hvplot(
        self,
        data: Optional[Any] = None,
        obs_col: Optional[str] = None,
        fcst_col: Optional[str] = None,
        n_bins: int = 10,
        label_col: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Generates an interactive plot using hvPlot.

        Parameters
        ----------
        data : Any, optional
            Input data, overrides self.data if provided.
        obs_col : str, optional
            Name of the observation variable.
        fcst_col : str, optional
            Name of the forecast variable.
        n_bins : int, optional
            Number of bins, by default 10.
        label_col : str, optional
            Column name to group by.
        **kwargs : Any
            Additional hvPlot arguments.

        Returns
        -------
        holoviews.core.Element
            The interactive plot.
        """
        try:
            import holoviews as hv
            import hvplot.xarray  # noqa: F401
        except ImportError:
            raise ImportError(
                "hvplot and holoviews are required for interactive plotting. Install them with 'pip install hvplot holoviews'."
            )

        plot_data = normalize_data(data) if data is not None else self.data
        if plot_data is None:
            raise ValueError("No data provided.")

        if label_col:

            def get_stats(group):
                return compute_binned_bias(
                    group[obs_col], group[fcst_col], n_bins=n_bins
                ).compute()

            # We compute per group for the visualization summary
            stats_list = []
            for name, group in plot_data.groupby(label_col):
                s = get_stats(group)
                s = s.assign_coords({label_col: name}).expand_dims(label_col)
                stats_list.append(s)
            pdf = xr.concat(stats_list, dim=label_col).dropna(dim="bin_center")
            by = label_col
        else:
            if isinstance(plot_data, xr.Dataset):
                obs = plot_data[obs_col]
                mod = plot_data[fcst_col]
            elif isinstance(plot_data, pd.DataFrame):
                obs = plot_data[obs_col]
                mod = plot_data[fcst_col]
            else:
                mod = plot_data
                obs = kwargs.pop("obs")
            pdf = compute_binned_bias(obs, mod, n_bins=n_bins).compute()
            pdf = pdf.dropna(dim="bin_center")
            by = None

        xlabel = (
            plot_data[obs_col].attrs.get("long_name", obs_col)
            if obs_col
            else "Observed Value"
        )

        plot = pdf.hvplot.scatter(
            x="bin_center",
            y="bias_mean",
            by=by,
            xlabel=xlabel,
            ylabel="Mean Bias",
            **kwargs,
        ) * pdf.hvplot.errorbars(x="bin_center", y="bias_mean", yerr1="bias_std", by=by)

        # Add zero line
        plot *= hv.HLine(0).opts(color="black", line_dash="dashed")

        return plot
