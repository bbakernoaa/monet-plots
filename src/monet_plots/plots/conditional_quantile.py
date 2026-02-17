# src/monet_plots/plots/conditional_quantile.py
"""Conditional quantile plot for model evaluation."""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from .base import BasePlot
from ..plot_utils import to_dataframe
from typing import Any, List, Union


class ConditionalQuantilePlot(BasePlot):
    """Conditional quantile plot.

    Plots the distribution (quantiles) of modeled values as a function
    of binned observed values. This helps identify if the model's
    uncertainty or bias changes across the range of observations.
    """

    def __init__(
        self,
        data: Any,
        obs_col: str,
        mod_col: str,
        *,
        bins: Union[int, List[float]] = 10,
        quantiles: List[float] = [0.25, 0.5, 0.75],
        **kwargs,
    ):
        """
        Initialize Conditional Quantile Plot.

        Args:
            data: Input data (DataFrame, DataArray, etc.).
            obs_col: Column name for observations.
            mod_col: Column name for model values.
            bins: Number of bins or bin edges for observations.
            quantiles: List of quantiles to calculate (0 to 1).
            **kwargs: Arguments passed to BasePlot.
        """
        super().__init__(**kwargs)
        self.df = to_dataframe(data).dropna(subset=[obs_col, mod_col])
        self.obs_col = obs_col
        self.mod_col = mod_col
        self.bins = bins
        self.quantiles = sorted(quantiles)

    def plot(self, show_points: bool = False, **kwargs):
        """Generate the conditional quantile plot."""
        # Bin observations
        self.df["bin"] = pd.cut(self.df[self.obs_col], bins=self.bins)

        # Calculate quantiles for each bin
        # We use the midpoint of each bin for the x-axis
        bin_midpoints = []
        quantile_vals = {q: [] for q in self.quantiles}

        for bin_name, group in self.df.groupby("bin", observed=True):
            bin_midpoints.append(bin_name.mid)
            for q in self.quantiles:
                quantile_vals[q].append(group[self.mod_col].quantile(q))

        # Plotting
        if show_points:
            self.ax.scatter(
                self.df[self.obs_col],
                self.df[self.mod_col],
                alpha=0.3,
                s=10,
                color="grey",
                label="Data",
            )

        # Plot 1:1 line
        lims = [
            min(self.df[self.obs_col].min(), self.df[self.mod_col].min()),
            max(self.df[self.obs_col].max(), self.df[self.mod_col].max()),
        ]
        self.ax.plot(lims, lims, "k--", alpha=0.5, label="1:1")

        # Plot quantiles
        colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(self.quantiles)))
        for i, q in enumerate(self.quantiles):
            label = f"{int(q * 100)}th percentile"
            linestyle = "-" if q == 0.5 else "--"
            linewidth = 2 if q == 0.5 else 1
            self.ax.plot(
                bin_midpoints,
                quantile_vals[q],
                label=label,
                color=colors[i],
                linestyle=linestyle,
                linewidth=linewidth,
            )

        # Shading between quantiles if there are at least 2 (e.g. 25th and 75th)
        if 0.25 in self.quantiles and 0.75 in self.quantiles:
            self.ax.fill_between(
                bin_midpoints,
                quantile_vals[0.25],
                quantile_vals[0.75],
                color="blue",
                alpha=0.1,
            )

        self.ax.set_xlabel(f"Observed: {self.obs_col}")
        self.ax.set_ylabel(f"Modeled: {self.mod_col}")
        self.ax.legend()
        self.ax.grid(True, linestyle=":", alpha=0.6)

        return self.ax

    def hvplot(self, **kwargs):
        """Generate an interactive conditional quantile plot using hvPlot."""
        import hvplot.pandas  # noqa: F401
        import holoviews as hv

        # Bin observations
        self.df["bin"] = pd.cut(self.df[self.obs_col], bins=self.bins)

        # Calculate quantiles for each bin
        bin_midpoints = []
        quantile_vals = {q: [] for q in self.quantiles}

        for bin_name, group in self.df.groupby("bin", observed=True):
            bin_midpoints.append(bin_name.mid)
            for q in self.quantiles:
                quantile_vals[q].append(group[self.mod_col].quantile(q))

        # Create DataFrame for hvplot
        plot_df = pd.DataFrame({"bin_midpoint": bin_midpoints})
        for q in self.quantiles:
            plot_df[f"q{q}"] = quantile_vals[q]

        plot_kwargs = {
            "x": "bin_midpoint",
            "y": [f"q{q}" for q in self.quantiles],
            "kind": "line",
            "xlabel": f"Observed: {self.obs_col}",
            "ylabel": f"Modeled: {self.mod_col}",
            "title": "Conditional Quantile",
        }
        plot_kwargs.update(kwargs)

        p = plot_df.hvplot(**plot_kwargs)

        # 1:1 line
        lims = [
            min(self.df[self.obs_col].min(), self.df[self.mod_col].min()),
            max(self.df[self.obs_col].max(), self.df[self.mod_col].max()),
        ]
        one_to_one = hv.Curve([(lims[0], lims[0]), (lims[1], lims[1])]).opts(
            color="black", alpha=0.5, line_dash="dashed"
        )

        return one_to_one * p
