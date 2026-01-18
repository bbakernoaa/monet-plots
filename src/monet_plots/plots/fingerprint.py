# src/monet_plots/plots/fingerprint.py
"""Fingerprint plot for visualizing temporal patterns."""

import pandas as pd
import seaborn as sns
from .base import BasePlot
from ..plot_utils import to_dataframe
from typing import Any


class FingerprintPlot(BasePlot):
    """Fingerprint plot.

    Displays a variable as a heatmap across two different temporal scales,
    such as hour of day vs. day of year, to reveal periodic patterns.
    """

    def __init__(
        self,
        data: Any,
        val_col: str,
        *,
        time_col: str = "time",
        x_scale: str = "hour",
        y_scale: str = "dayofyear",
        **kwargs,
    ):
        """
        Initialize Fingerprint Plot.

        Args:
            data: Input data (DataFrame, DataArray, etc.).
            val_col: Column name of the value to plot.
            time_col: Column name for timestamp.
            x_scale: Temporal scale for the x-axis ('hour', 'month', 'dayofweek', etc.).
            y_scale: Temporal scale for the y-axis ('dayofyear', 'year', 'week', etc.).
            **kwargs: Arguments passed to BasePlot.
        """
        super().__init__(**kwargs)
        self.df = to_dataframe(data).copy()
        self.val_col = val_col
        self.time_col = time_col
        self.x_scale = x_scale
        self.y_scale = y_scale

        # Ensure time_col is datetime
        self.df[self.time_col] = pd.to_datetime(self.df[self.time_col])

        self._extract_scale(self.x_scale, "x_val")
        self._extract_scale(self.y_scale, "y_val")

    def _extract_scale(self, scale: str, target_col: str):
        """Extract temporal features from datetime."""
        t = self.df[self.time_col].dt
        if scale == "hour":
            self.df[target_col] = t.hour
        elif scale == "month":
            self.df[target_col] = t.month
        elif scale == "dayofweek":
            self.df[target_col] = t.dayofweek
        elif scale == "dayofyear":
            self.df[target_col] = t.dayofyear
        elif scale == "week":
            self.df[target_col] = t.isocalendar().week
        elif scale == "year":
            self.df[target_col] = t.year
        elif scale == "date":
            self.df[target_col] = t.date
        else:
             # Try to use it as a direct column if not a known scale
             if scale in self.df.columns:
                 self.df[target_col] = self.df[scale]
             else:
                 raise ValueError(f"Unknown temporal scale: {scale}")

    def plot(self, cmap: str = "viridis", **kwargs):
        """Generate the fingerprint heatmap."""
        pivot_df = self.df.pivot_table(
            index="y_val", columns="x_val", values=self.val_col, aggfunc="mean"
        )

        sns.heatmap(pivot_df, ax=self.ax, cmap=cmap, **kwargs)

        self.ax.set_xlabel(self.x_scale.capitalize())
        self.ax.set_ylabel(self.y_scale.capitalize())
        self.ax.set_title(f"Fingerprint: {self.val_col}")

        return self.ax
