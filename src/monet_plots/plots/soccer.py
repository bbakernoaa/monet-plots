# src/monet_plots/plots/soccer.py
"""Soccer plot for model evaluation."""

import matplotlib.patches as patches
import numpy as np
import xarray as xr
from .base import BasePlot
from ..plot_utils import to_dataframe
from typing import Any, Optional, Dict


class SoccerPlot(BasePlot):
    """Soccer plot for model evaluation.

    This plot shows model performance by plotting bias (x-axis) against error (y-axis).
    It typically includes 'goal' and 'criteria' zones to visually assess if the
    model meets specific performance standards.
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
        **kwargs: Any,
    ) -> None:
        """Initialize Soccer Plot.

        Parameters
        ----------
        data : Any
            Input data (DataFrame, DataArray, etc.).
        obs_col : str, optional
            Column name for observations. Required if bias/error not provided.
        mod_col : str, optional
            Column name for model values. Required if bias/error not provided.
        bias_col : str, optional
            Column name for pre-calculated bias.
        error_col : str, optional
            Column name for pre-calculated error.
        label_col : str, optional
            Column name for labeling points.
        metric : str, optional
            Type of metric to calculate if obs/mod provided ('fractional' or 'normalized').
        goal : dict, optional
            Dictionary with 'bias' and 'error' thresholds for the goal zone.
        criteria : dict, optional
            Dictionary with 'bias' and 'error' thresholds for the criteria zone.
        **kwargs : Any
            Arguments passed to BasePlot.

        Examples
        --------
        >>> plot = SoccerPlot(ds, obs_col="obs", mod_col="mod")
        """
        super().__init__(**kwargs)

        if isinstance(data, (xr.DataArray, xr.Dataset)):
            self.data = data
        else:
            self.data = to_dataframe(data)

        self.bias_col = bias_col
        self.error_col = error_col
        self.label_col = label_col
        self.metric = metric
        self.goal = goal
        self.criteria = criteria

        if bias_col is None or error_col is None:
            if obs_col is None or mod_col is None:
                raise ValueError(
                    "Must provide either bias_col/error_col or obs_col/mod_col"
                )
            self._calculate_metrics(obs_col, mod_col)
        else:
            self.bias_data = self.data[bias_col]
            self.error_data = self.data[error_col]

    def _calculate_metrics(self, obs_col: str, mod_col: str) -> None:
        """Calculate MFB/MFE or NMB/NME. Preserves Xarray/Dask laziness.

        Parameters
        ----------
        obs_col : str
            Column name for observations.
        mod_col : str
            Column name for model values.

        Examples
        --------
        >>> plot._calculate_metrics("obs", "mod")
        """
        obs = self.data[obs_col]
        mod = self.data[mod_col]

        if isinstance(self.data, (xr.DataArray, xr.Dataset)):
            if self.metric == "fractional":
                denom = obs + mod
                self.bias_data = (200.0 * (mod - obs) / denom).where(denom != 0, np.nan)
                self.error_data = (200.0 * np.abs(mod - obs) / denom).where(
                    denom != 0, np.nan
                )
                self.xlabel = "Mean Fractional Bias (%)"
                self.ylabel = "Mean Fractional Error (%)"
            elif self.metric == "normalized":
                self.bias_data = (100.0 * (mod - obs) / obs).where(obs != 0, np.nan)
                self.error_data = (100.0 * np.abs(mod - obs) / obs).where(
                    obs != 0, np.nan
                )
                self.xlabel = "Normalized Mean Bias (%)"
                self.ylabel = "Normalized Mean Error (%)"
            else:
                raise ValueError("metric must be 'fractional' or 'normalized'")
        else:
            # Pandas/NumPy fallback
            if self.metric == "fractional":
                denom = (obs + mod).astype(float)
                self.bias_data = np.divide(
                    200.0 * (mod - obs),
                    denom,
                    out=np.full(denom.shape, np.nan),
                    where=denom != 0,
                )
                self.error_data = np.divide(
                    200.0 * np.abs(mod - obs),
                    denom,
                    out=np.full(denom.shape, np.nan),
                    where=denom != 0,
                )
                self.xlabel = "Mean Fractional Bias (%)"
                self.ylabel = "Mean Fractional Error (%)"
            elif self.metric == "normalized":
                obs_float = obs.astype(float)
                self.bias_data = np.divide(
                    100.0 * (mod - obs),
                    obs_float,
                    out=np.full(obs_float.shape, np.nan),
                    where=obs_float != 0,
                )
                self.error_data = np.divide(
                    100.0 * np.abs(mod - obs),
                    obs_float,
                    out=np.full(obs_float.shape, np.nan),
                    where=obs_float != 0,
                )
                self.xlabel = "Normalized Mean Bias (%)"
                self.ylabel = "Normalized Mean Error (%)"
            else:
                raise ValueError("metric must be 'fractional' or 'normalized'")

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

        # Plot points
        scatter_kwargs = {"zorder": 5}
        scatter_kwargs.update(kwargs)
        # Matplotlib will trigger computation if dask-backed
        self.ax.scatter(self.bias_data, self.error_data, **scatter_kwargs)

        # Labels
        if self.label_col is not None:
            labels = self.data[self.label_col]
            # Ensure we have concrete values for annotation
            if hasattr(labels, "compute"):
                labels = labels.compute()
            b_vals = (
                self.bias_data.compute()
                if hasattr(self.bias_data, "compute")
                else self.bias_data
            )
            e_vals = (
                self.error_data.compute()
                if hasattr(self.error_data, "compute")
                else self.error_data
            )

            for i, txt in enumerate(labels):
                self.ax.annotate(
                    txt,
                    (
                        b_vals.iloc[i] if hasattr(b_vals, "iloc") else b_vals[i],
                        e_vals.iloc[i] if hasattr(e_vals, "iloc") else e_vals[i],
                    ),
                    xytext=(5, 5),
                    textcoords="offset points",
                )

        # Setup axes
        limit = 0
        if self.criteria:
            limit = max(limit, self.criteria["bias"] * 1.1)
            limit_y = self.criteria["error"] * 1.1
        else:
            limit = max(limit, float(self.bias_data.abs().max()) * 1.1)
            limit_y = float(self.error_data.max()) * 1.1

        self.ax.set_xlim(-limit, limit)
        self.ax.set_ylim(0, limit_y)

        self.ax.axvline(0, color="k", linestyle="--", alpha=0.5)
        self.ax.set_xlabel(getattr(self, "xlabel", "Bias (%)"))
        self.ax.set_ylabel(getattr(self, "ylabel", "Error (%)"))
        self.ax.grid(True, linestyle=":", alpha=0.6)

        return self.ax
