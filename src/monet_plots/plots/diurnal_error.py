# src/monet_plots/plots/diurnal_error.py
"""Diurnal error heat map for model performance analysis."""

import pandas as pd
import numpy as np
import seaborn as sns
from .base import BasePlot
from ..plot_utils import to_dataframe
from typing import Any


class DiurnalErrorPlot(BasePlot):
    """Diurnal error heat map.

    Visualizes model error (bias) as a function of the hour of day and another
    temporal dimension (e.g., month, day of week, or date).
    """

    def __init__(
        self,
        data: Any,
        obs_col: str,
        mod_col: str,
        *,
        time_col: str = "time",
        second_dim: str = "month",
        metric: str = "bias",
        **kwargs,
    ):
        """
        Initialize Diurnal Error Plot.

        Args:
            data: Input data (DataFrame, DataArray, etc.).
            obs_col: Column name for observations.
            mod_col: Column name for model values.
            time_col: Column name for timestamp.
            second_dim: The second dimension for the heatmap ('month', 'dayofweek', or 'date').
            metric: The metric to plot ('bias' or 'error').
            **kwargs: Arguments passed to BasePlot.
        """
        super().__init__(**kwargs)
        self.df = to_dataframe(data).copy()
        self.obs_col = obs_col
        self.mod_col = mod_col
        self.time_col = time_col
        self.second_dim = second_dim
        self.metric = metric

        # Ensure time_col is datetime
        self.df[self.time_col] = pd.to_datetime(self.df[self.time_col])
        self.df["hour"] = self.df[self.time_col].dt.hour

        if self.second_dim == "month":
            self.df["second_val"] = self.df[self.time_col].dt.month
            self.second_label = "Month"
        elif self.second_dim == "dayofweek":
            self.df["second_val"] = self.df[self.time_col].dt.dayofweek
            self.second_label = "Day of Week"
        elif self.second_dim == "date":
            self.df["second_val"] = self.df[self.time_col].dt.date
            self.second_label = "Date"
        else:
            # Assume it's a column name already in df
            self.df["second_val"] = self.df[self.second_dim]
            self.second_label = self.second_dim

        # Calculate error/bias
        if self.metric == "bias":
            self.df["val"] = self.df[self.mod_col] - self.df[self.obs_col]
        elif self.metric == "error":
            self.df["val"] = np.abs(self.df[self.mod_col] - self.df[self.obs_col])
        else:
            raise ValueError("metric must be 'bias' or 'error'")

    def plot(self, cmap: str = "RdBu_r", **kwargs):
        """Generate the diurnal error heatmap."""
        pivot_df = self.df.pivot_table(
            index="second_val", columns="hour", values="val", aggfunc="mean"
        )

        sns.heatmap(
            pivot_df,
            ax=self.ax,
            cmap=cmap,
            center=0 if self.metric == "bias" else None,
            **kwargs,
        )

        self.ax.set_xlabel("Hour of Day")
        self.ax.set_ylabel(self.second_label)
        self.ax.set_title(f"Diurnal {self.metric.capitalize()}")

        return self.ax
