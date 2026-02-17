# src/monet_plots/plots/soccer.py
"""Soccer plot for model evaluation."""

from __future__ import annotations

import matplotlib.patches as patches
import numpy as np
import pandas as pd
import xarray as xr
from .base import BasePlot
from ..plot_utils import normalize_data
from ..verification_metrics import compute_fb, compute_fe, compute_nmb, compute_nme
from typing import Any, Optional, Dict


class SoccerPlot(BasePlot):
    """Soccer plot for model evaluation.

    This plot shows model performance by plotting bias (x-axis) against error (y-axis).
    It typically includes 'goal' and 'criteria' zones to visually assess if the
    model meets specific performance standards.

    Attributes
    ----------
    data : Union[pd.DataFrame, xr.Dataset, xr.DataArray]
        The input data for the plot.
    bias_data : Union[pd.Series, xr.DataArray]
        Calculated or provided bias values.
    error_data : Union[pd.Series, xr.DataArray]
        Calculated or provided error values.
    """

    def __init__(
        self,
        data: Any,
        *,
        obs_col: Optional[str] = None,
        mod_col: Optional[str] = None,
        bias_col: Optional[str] = None,
        error_col: Optional[str] = None,
        label_col: Optional[str] = None,
        metric: str = "fractional",
        goal: Optional[Dict[str, float]] = {"bias": 30.0, "error": 50.0},
        criteria: Optional[Dict[str, float]] = {"bias": 60.0, "error": 75.0},
        fig: Optional[matplotlib.figure.Figure] = None,
        ax: Optional[matplotlib.axes.Axes] = None,
        **kwargs: Any,
    ):
        """
        Initialize Soccer Plot.

        Args:
            data: Input data (DataFrame, DataArray, Dataset).
            obs_col: Column name for observations. Required if bias/error not provided.
            mod_col: Column name for model values. Required if bias/error not provided.
            bias_col: Column name for pre-calculated bias.
            error_col: Column name for pre-calculated error.
            label_col: Column name for labeling points.
            metric: Type of metric to calculate if obs/mod provided ('fractional' or 'normalized').
            goal: Dictionary with 'bias' and 'error' thresholds for the goal zone.
            criteria: Dictionary with 'bias' and 'error' thresholds for the criteria zone.
            **kwargs: Arguments passed to BasePlot.
        """
        super().__init__(**kwargs)
        self.data = normalize_data(data)
        self.bias_col = bias_col
        self.error_col = error_col
        self.label_col = label_col
        self.metric = metric
        self.goal = goal
        self.criteria = criteria

        # Track provenance for Xarray
        if isinstance(self.data, (xr.DataArray, xr.Dataset)):
            history = self.data.attrs.get("history", "")
            self.data.attrs["history"] = f"Initialized SoccerPlot; {history}"

        if bias_col is None or error_col is None:
            if obs_col is None or mod_col is None:
                raise ValueError(
                    "Must provide either bias_col/error_col or obs_col/mod_col"
                )
            self._calculate_metrics(obs_col, mod_col)
        else:
            self.bias_data = self.data[bias_col]
            self.error_data = self.data[error_col]

    def _calculate_metrics(self, obs_col: str, mod_col: str):
        """Calculate MFB/MFE or NMB/NME preserving granularity."""
        obs = self.data[obs_col]
        mod = self.data[mod_col]

        if isinstance(self.data, (xr.DataArray, xr.Dataset)):
            # Use dim=[] to maintain per-sample granularity (no aggregation)
            dim_arg = []
        else:
            dim_arg = ()

        if self.metric == "fractional":
            self.bias_data = compute_fb(obs, mod, dim=dim_arg)
            self.error_data = compute_fe(obs, mod, dim=dim_arg)
            self.xlabel = "Mean Fractional Bias (%)"
            self.ylabel = "Mean Fractional Error (%)"

        elif self.metric == "normalized":
            self.bias_data = compute_nmb(obs, mod, dim=dim_arg)
            self.error_data = compute_nme(obs, mod, dim=dim_arg)
            self.xlabel = "Normalized Mean Bias (%)"
            self.ylabel = "Normalized Mean Error (%)"
        else:
            raise ValueError("metric must be 'fractional' or 'normalized'")

        # Ensure consistent output types for DataFrame inputs
        if isinstance(self.data, pd.DataFrame):
            self.bias_data = pd.Series(self.bias_data, index=self.data.index)
            self.error_data = pd.Series(self.error_data, index=self.data.index)

    def plot(self, **kwargs):
        """Generate the soccer plot."""
        # Draw zones
        if self.criteria:
            rect_crit = patches.Rectangle(
                (-self.criteria["bias"], 0),
                2 * self.criteria["bias"],
                self.criteria["error"],
                linewidth=1,
                edgecolor="lightgrey",
                facecolor="lightgrey",
                alpha=0.3,
                label="Criteria",
                zorder=0,
            )
            self.ax.add_patch(rect_crit)

        if self.goal:
            rect_goal = patches.Rectangle(
                (-self.goal["bias"], 0),
                2 * self.goal["bias"],
                self.goal["error"],
                linewidth=1,
                edgecolor="grey",
                facecolor="grey",
                alpha=0.3,
                label="Goal",
                zorder=1,
            )
            self.ax.add_patch(rect_goal)

        # Plot points - compute if lazy
        bias = self.bias_data
        error = self.error_data

        if hasattr(bias, "compute"):
            bias = bias.compute()
        if hasattr(error, "compute"):
            error = error.compute()

        scatter_kwargs = {"zorder": 5}
        scatter_kwargs.update(kwargs)
        self.ax.scatter(bias, error, **scatter_kwargs)

        # Labels
        if self.label_col is not None:
            labels = (
                self.data[self.label_col].values
                if isinstance(self.data, (xr.DataArray, xr.Dataset))
                else self.data[self.label_col]
            )
            bias_vals = (
                self.bias_data.values
                if hasattr(self.bias_data, "values")
                else self.bias_data
            )
            error_vals = (
                self.error_data.values
                if hasattr(self.error_data, "values")
                else self.error_data
            )
            for i, txt in enumerate(labels):
                self.ax.annotate(
                    txt,
                    (bias_vals[i], error_vals[i]),
                    xytext=(5, 5),
                    textcoords="offset points",
                )

        # Setup axes - use compute() for dask objects to get limits
        bias_max = np.abs(self.bias_data).max()
        error_max = self.error_data.max()
        if hasattr(bias_max, "compute"):
            import dask

            bias_max, error_max = dask.compute(bias_max, error_max)

        limit = 0
        if self.criteria:
            limit = max(limit, self.criteria["bias"] * 1.1)
            limit_y = self.criteria["error"] * 1.1
        else:
            limit = max(limit, float(bias_max) * 1.1)
            limit_y = float(error_max) * 1.1

        self.ax.set_xlim(-limit, limit)
        self.ax.set_ylim(0, limit_y)

        self.ax.axvline(0, color="k", linestyle="--", alpha=0.5)
        self.ax.set_xlabel(getattr(self, "xlabel", "Bias (%)"))
        self.ax.set_ylabel(getattr(self, "ylabel", "Error (%)"))
        self.ax.grid(True, linestyle=":", alpha=0.6)

        # Update history for provenance
        if isinstance(self.data, (xr.DataArray, xr.Dataset)):
            history = self.data.attrs.get("history", "")
            self.data.attrs["history"] = f"Generated SoccerPlot; {history}"

        return self.ax

    def hvplot(self, **kwargs):
        """Generate an interactive soccer plot using hvPlot."""
        import hvplot.pandas  # noqa: F401
        import holoviews as hv

        df_soccer = pd.DataFrame(
            {"bias": self.bias_data, "error": self.error_data}
        ).reset_index(drop=True)
        if self.label_col is not None:
            labels = (
                self.data[self.label_col].values
                if isinstance(self.data, (xr.DataArray, xr.Dataset))
                else self.data[self.label_col]
            )
            df_soccer["label"] = labels

        plot_kwargs = {
            "x": "bias",
            "y": "error",
            "kind": "scatter",
            "xlabel": getattr(self, "xlabel", "Bias (%)"),
            "ylabel": getattr(self, "ylabel", "Error (%)"),
            "title": "Soccer Plot",
        }
        if "label" in df_soccer.columns:
            plot_kwargs["hover_cols"] = ["label"]

        plot_kwargs.update(kwargs)

        p = df_soccer.hvplot(**plot_kwargs)

        # Add zones
        zones = []
        if self.criteria:
            zones.append(
                hv.Rectangles(
                    [
                        (
                            -self.criteria["bias"],
                            0,
                            self.criteria["bias"],
                            self.criteria["error"],
                        )
                    ]
                ).opts(alpha=0.2, color="lightgrey", title="Criteria")
            )
        if self.goal:
            zones.append(
                hv.Rectangles(
                    [(-self.goal["bias"], 0, self.goal["bias"], self.goal["error"])]
                ).opts(alpha=0.2, color="grey", title="Goal")
            )

        if zones:
            return hv.Overlay(zones) * p
        return p
