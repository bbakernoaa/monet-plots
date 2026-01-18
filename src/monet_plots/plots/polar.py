# src/monet_plots/plots/polar.py
"""Bivariate polar plot for analyzing variable dependence on wind."""

import numpy as np
from .base import BasePlot
from ..plot_utils import to_dataframe
from typing import Any, Optional


class BivariatePolarPlot(BasePlot):
    """Bivariate polar plot.

    Shows how a variable varies with wind speed and wind direction.
    Uses polar coordinates where the angle represents wind direction
    and the radius represents wind speed.
    """

    def __init__(
        self,
        data: Any,
        ws_col: str,
        wd_col: str,
        val_col: str,
        *,
        ws_max: Optional[float] = None,
        **kwargs,
    ):
        """
        Initialize Bivariate Polar Plot.

        Args:
            data: Input data (DataFrame, DataArray, etc.).
            ws_col: Column name for wind speed.
            wd_col: Column name for wind direction (degrees).
            val_col: Column name for the value to plot.
            ws_max: Maximum wind speed to show.
            **kwargs: Arguments passed to BasePlot. Note: 'subplot_kw={"projection": "polar"}'
                      is added automatically if not provided.
        """
        if "subplot_kw" not in kwargs:
            kwargs["subplot_kw"] = {"projection": "polar"}
        elif "projection" not in kwargs["subplot_kw"]:
             kwargs["subplot_kw"]["projection"] = "polar"

        super().__init__(**kwargs)
        self.df = to_dataframe(data).dropna(subset=[ws_col, wd_col, val_col])
        self.ws_col = ws_col
        self.wd_col = wd_col
        self.val_col = val_col
        self.ws_max = ws_max or self.df[ws_col].max()

    def plot(self, n_bins_ws: int = 10, n_bins_wd: int = 36, cmap: str = "viridis", **kwargs):
        """
        Generate the bivariate polar plot.

        Uses binning to aggregate data before plotting.
        """
        # Convert wind direction to radians and adjust for polar plot (0 is North/Up)
        # Matplotlib polar 0 is East (right). We want 0 at North.
        # Wind direction is usually 0=North, 90=East.
        # theta = (90 - wd) * pi / 180
        theta_rad = np.radians(self.df[self.wd_col])
        # Matplotlib's polar axis by default has 0 at East.
        # To make 0 North, we can use:
        self.ax.set_theta_zero_location("N")
        self.ax.set_theta_direction(-1) # Clockwise

        # Binning
        ws_bins = np.linspace(0, self.ws_max, n_bins_ws + 1)
        wd_bins = np.radians(np.linspace(0, 360, n_bins_wd + 1))

        # We can use np.histogram2d
        # Note: wd_bins is in radians
        H, xedges, yedges = np.histogram2d(
            theta_rad, self.df[self.ws_col],
            bins=[wd_bins, ws_bins],
            weights=self.df[self.val_col]
        )
        Counts, _, _ = np.histogram2d(
            theta_rad, self.df[self.ws_col],
            bins=[wd_bins, ws_bins]
        )

        # Calculate mean
        with np.errstate(divide='ignore', invalid='ignore'):
            Z = H / Counts

        # Meshgrid for plotting
        # np.histogram2d edges are for pcolormesh
        Theta, R = np.meshgrid(wd_bins, ws_bins)

        # Plotting
        mappable = self.ax.pcolormesh(Theta, R, Z.T, cmap=cmap, **kwargs)
        self.fig.colorbar(mappable, ax=self.ax, label=self.val_col)

        self.ax.set_ylim(0, self.ws_max)
        return self.ax
