# src/monet_plots/plots/soccer.py
"""Soccer plot for model evaluation."""
from __future__ import annotations

import matplotlib.patches as patches
import numpy as np
import pandas as pd
import xarray as xr
from typing import Any, Optional, Dict, Union, TYPE_CHECKING

from .base import BasePlot
from ..plot_utils import normalize_data

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure


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

        Parameters
        ----------
        data : Any
            Input data. Can be a pandas DataFrame, xarray DataArray,
            xarray Dataset, or numpy ndarray.
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
            Type of metric to calculate if obs/mod provided ('fractional' or 'normalized'),
            by default "fractional".
        goal : Dict[str, float], optional
            Dictionary with 'bias' and 'error' thresholds for the goal zone,
            by default {"bias": 30.0, "error": 50.0}.
        criteria : Dict[str, float], optional
            Dictionary with 'bias' and 'error' thresholds for the criteria zone,
            by default {"bias": 60.0, "error": 75.0}.
        fig : matplotlib.figure.Figure, optional
            An existing Figure object.
        ax : matplotlib.axes.Axes, optional
            An existing Axes object.
        **kwargs : Any
            Arguments passed to BasePlot.
        """
        super().__init__(fig=fig, ax=ax, **kwargs)
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

    def _calculate_metrics(self, obs_col: str, mod_col: str) -> None:
        """Calculate MFB/MFE or NMB/NME using vectorized operations.

        Parameters
        ----------
        obs_col : str
            Column/variable name for observations.
        mod_col : str
            Column/variable name for model values.
        """
        obs = self.data[obs_col]
        mod = self.data[mod_col]

        if self.metric == "fractional":
            # Mean Fractional Bias and Error
            denom = (obs + mod).astype(float)
            num_bias = 200.0 * (mod - obs)
            num_error = 200.0 * np.abs(mod - obs)

            if isinstance(denom, xr.DataArray):
                self.bias_data = (num_bias / denom).where(denom != 0, np.nan)
                self.error_data = (num_error / denom).where(denom != 0, np.nan)
            else:
                self.bias_data = np.divide(
                    num_bias, denom, out=np.full(denom.shape, np.nan), where=denom != 0
                )
                self.error_data = np.divide(
                    num_error, denom, out=np.full(denom.shape, np.nan), where=denom != 0
                )

            self.xlabel = "Mean Fractional Bias (%)"
            self.ylabel = "Mean Fractional Error (%)"

        elif self.metric == "normalized":
            # Normalized Mean Bias and Error
            obs_float = obs.astype(float)
            num_bias = 100.0 * (mod - obs)
            num_error = 100.0 * np.abs(mod - obs)

            if isinstance(obs_float, xr.DataArray):
                self.bias_data = (num_bias / obs_float).where(obs_float != 0, np.nan)
                self.error_data = (num_error / obs_float).where(obs_float != 0, np.nan)
            else:
                self.bias_data = np.divide(
                    num_bias,
                    obs_float,
                    out=np.full(obs_float.shape, np.nan),
                    where=obs_float != 0,
                )
                self.error_data = np.divide(
                    num_error,
                    obs_float,
                    out=np.full(obs_float.shape, np.nan),
                    where=obs_float != 0,
                )

            self.xlabel = "Normalized Mean Bias (%)"
            self.ylabel = "Normalized Mean Error (%)"
        else:
            raise ValueError("metric must be 'fractional' or 'normalized'")

        # Update history if Xarray
        if isinstance(self.bias_data, xr.DataArray):
            history = self.bias_data.attrs.get("history", "")
            self.bias_data.attrs["history"] = f"Calculated {self.metric} soccer metrics; {history}"

    def plot(self, **kwargs: Any) -> matplotlib.axes.Axes:
        """Generate the soccer plot.

        Parameters
        ----------
        **kwargs : Any
            Keyword arguments passed to `ax.scatter`.

        Returns
        -------
        matplotlib.axes.Axes
            The axes object with the soccer plot.
        """
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
            labels = self.data[self.label_col]
            if hasattr(labels, "values"):
                labels = labels.values

            for i, txt in enumerate(labels):
                # Ensure we have scalar values for annotation
                b_val = bias.iloc[i] if hasattr(bias, "iloc") else bias[i]
                e_val = error.iloc[i] if hasattr(error, "iloc") else error[i]
                self.ax.annotate(
                    str(txt),
                    (b_val, e_val),
                    xytext=(5, 5),
                    textcoords="offset points",
                )

        # Setup axes
        limit = 0
        if self.criteria:
            limit = max(limit, self.criteria["bias"] * 1.1)
            limit_y = self.criteria["error"] * 1.1
        else:
            limit = max(limit, float(np.abs(bias).max()) * 1.1)
            limit_y = float(error.max()) * 1.1

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
