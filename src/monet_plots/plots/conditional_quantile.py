# src/monet_plots/plots/conditional_quantile.py
"""Conditional quantile plot for model evaluation."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from ..plot_utils import normalize_data
from ..verification_metrics import compute_binned_quantiles
from .base import BasePlot


class ConditionalQuantilePlot(BasePlot):
    """Conditional quantile plot.

    Plots the distribution (quantiles) of modeled values as a function
    of binned observed values. This helps identify if the model's
    uncertainty or bias changes across the range of observations.
    """

    def __init__(
        self,
        data: Any | None = None,
        obs_col: str | None = None,
        mod_col: str | None = None,
        *,
        bins: int | list[float] = 10,
        quantiles: list[float] = [0.25, 0.5, 0.75],
        fig: Any | None = None,
        ax: Any | None = None,
        **kwargs: Any,
    ):
        """
        Initialize Conditional Quantile Plot.

        Parameters
        ----------
        data : Any, optional
            Input data (DataFrame, DataArray, Dataset, or ndarray).
        obs_col : str, optional
            Column name for observations. Required for Dataset/DataFrame.
        mod_col : str, optional
            Column name for model values. Required for Dataset/DataFrame.
        bins : int | list[float], optional
            Number of bins or bin edges for observations, by default 10.
        quantiles : list[float], optional
            List of quantiles to calculate (0 to 1), by default [0.25, 0.5, 0.75].
        fig : matplotlib.figure.Figure, optional
            Figure to plot on.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.
        **kwargs : Any
            Additional keyword arguments passed to BasePlot.
        """
        super().__init__(fig=fig, ax=ax, **kwargs)
        self.data = normalize_data(data) if data is not None else None
        self.obs_col = obs_col
        self.mod_col = mod_col
        self.bins = bins
        self.quantiles = sorted(quantiles)

    def plot(
        self,
        data: Any | None = None,
        obs_col: str | None = None,
        mod_col: str | None = None,
        show_points: bool = False,
        **kwargs: Any,
    ) -> plt.Axes:
        """
        Generate the conditional quantile plot.

        Parameters
        ----------
        data : Any, optional
            Input data, overrides self.data if provided.
        obs_col : str, optional
            Name of the observation variable.
        mod_col : str, optional
            Name of the model variable.
        show_points : bool, optional
            Whether to show raw data points, by default False.
        **kwargs : Any
            Additional plotting arguments for `ax.plot`.

        Returns
        -------
        matplotlib.axes.Axes
            The axes object.

        Examples
        --------
        >>> plot = ConditionalQuantilePlot(df, obs_col="obs", mod_col="mod")
        >>> ax = plot.plot(show_points=True)
        """
        plot_data = normalize_data(data) if data is not None else self.data
        if plot_data is None:
            raise ValueError("No data provided.")

        obs_var = obs_col or self.obs_col
        mod_var = mod_col or self.mod_col

        # Extract data arrays for computation
        if isinstance(plot_data, xr.Dataset):
            obs = plot_data[obs_var]
            mod = plot_data[mod_var]
        elif isinstance(plot_data, pd.DataFrame):
            obs = xr.DataArray(plot_data[obs_var], dims="index")
            mod = xr.DataArray(plot_data[mod_var], dims="index")
        elif isinstance(plot_data, xr.DataArray):
            mod = plot_data
            if obs_var is None:
                raise ValueError("obs_col must be provided if data is a DataArray.")
            obs = plot_data.coords[obs_var]
        else:
            raise TypeError(f"Unsupported data type: {type(plot_data)}")

        # Calculate binned quantiles using vectorized utility
        stats = compute_binned_quantiles(
            obs, mod, quantiles=self.quantiles, n_bins=self.bins
        )
        pdf = stats.compute().dropna(dim="bin_center")

        # Plotting
        if show_points:
            # Convert to numpy for scatter
            self.ax.scatter(
                obs.values.ravel(),
                mod.values.ravel(),
                alpha=0.3,
                s=10,
                color="grey",
                label="Data",
            )

        # Plot 1:1 line
        if hasattr(obs.data, "chunks") or hasattr(mod.data, "chunks"):
            import dask

            vmin, vmax = dask.compute(
                min(obs.min(), mod.min()), max(obs.max(), mod.max())
            )
        else:
            vmin = float(min(obs.min(), mod.min()))
            vmax = float(max(obs.max(), mod.max()))

        lims = [vmin, vmax]
        self.ax.plot(lims, lims, "k--", alpha=0.5, label="1:1")

        # Plot quantiles
        colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(self.quantiles)))
        for i, q in enumerate(self.quantiles):
            label = f"{int(q * 100)}th percentile"
            linestyle = "-" if q == 0.5 else "--"
            linewidth = 2 if q == 0.5 else 1
            # Select specific quantile from the dataset
            q_data = pdf.mod_quantile.sel(quantile=q)
            self.ax.plot(
                pdf.bin_center,
                q_data,
                label=label,
                color=colors[i],
                linestyle=linestyle,
                linewidth=linewidth,
                **kwargs,
            )

        # Shading between quantiles if there are at least 2 (e.g. 25th and 75th)
        if 0.25 in self.quantiles and 0.75 in self.quantiles:
            self.ax.fill_between(
                pdf.bin_center,
                pdf.mod_quantile.sel(quantile=0.25),
                pdf.mod_quantile.sel(quantile=0.75),
                color="blue",
                alpha=0.1,
            )

        xlabel = obs.attrs.get("long_name", obs_var)
        ylabel = mod.attrs.get("long_name", mod_var)

        self.ax.set_xlabel(f"Observed: {xlabel}")
        self.ax.set_ylabel(f"Modeled: {ylabel}")
        self.ax.legend()
        self.ax.grid(True, linestyle=":", alpha=0.6)

        return self.ax

    def hvplot(
        self,
        data: Any | None = None,
        obs_col: str | None = None,
        mod_col: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Generate an interactive plot using hvPlot.

        Parameters
        ----------
        data : Any, optional
            Input data, overrides self.data if provided.
        obs_col : str, optional
            Name of the observation variable.
        mod_col : str, optional
            Name of the model variable.
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

        obs_var = obs_col or self.obs_col
        mod_var = mod_col or self.mod_col

        # Extract data arrays
        if isinstance(plot_data, xr.Dataset):
            obs = plot_data[obs_var]
            mod = plot_data[mod_var]
        elif isinstance(plot_data, pd.DataFrame):
            obs = xr.DataArray(plot_data[obs_var], dims="index")
            mod = xr.DataArray(plot_data[mod_var], dims="index")
        elif isinstance(plot_data, xr.DataArray):
            mod = plot_data
            obs = plot_data.coords[obs_var]
        else:
            raise TypeError(f"Unsupported data type: {type(plot_data)}")

        stats = compute_binned_quantiles(
            obs, mod, quantiles=self.quantiles, n_bins=self.bins
        )
        pdf = stats.compute().dropna(dim="bin_center")

        xlabel = obs.attrs.get("long_name", obs_var)
        ylabel = mod.attrs.get("long_name", mod_var)

        # Plot quantiles
        # We need to reshape for hvplot if we want to use 'by' on quantiles
        plot = pdf.mod_quantile.hvplot.line(
            x="bin_center",
            by="quantile",
            xlabel=f"Observed: {xlabel}",
            ylabel=f"Modeled: {ylabel}",
            title="Conditional Quantiles",
            **kwargs,
        )

        # Add 1:1 line
        vmin = float(min(obs.min(), mod.min()))
        vmax = float(max(obs.max(), mod.max()))
        plot *= hv.Curve([[vmin, vmin], [vmax, vmax]], label="1:1").opts(
            color="black", line_dash="dashed"
        )

        return plot
