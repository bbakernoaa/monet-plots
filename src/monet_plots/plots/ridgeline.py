# src/monet_plots/plots/ridgeline.py
"""Ridgeline (joyplot) plot implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.stats import gaussian_kde

from ..colorbars import get_linear_scale
from ..plot_utils import normalize_data
from .base import BasePlot

if TYPE_CHECKING:
    import matplotlib.axes


class RidgelinePlot(BasePlot):
    """
    Creates a ridgeline plot (joyplot) from an xarray DataArray or pandas DataFrame.

    A ridgeline plot shows the distribution of a numeric value for several groups.
    Each group has its own distribution curve, often overlapping with others.

    Attributes:
        data (xr.DataArray | xr.Dataset | pd.DataFrame): Normalized input data.
        group_dim (str): The dimension or column to group by for the Y-axis.
        x (str | None): The column name for values if data is a DataFrame or Dataset.
        x_range (tuple | None): Tuple (min, max) for the x-axis limits.
        scale_factor (float): Height scaling of the curves.
        overlap (float): Vertical spacing between curves.
        cmap_name (str): Colormap name for coloring curves.
        title (str | None): Plot title.
    """

    def __init__(
        self,
        data: Any,
        group_dim: str,
        x: Optional[str] = None,
        *,
        x_range: Optional[Tuple[float, float]] = None,
        scale_factor: float = 1.0,
        overlap: float = 0.5,
        cmap: str = "RdBu_r",
        title: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Initializes the ridgeline plot with data and settings.

        Args:
            data (Any): The data to plot (xr.DataArray, xr.Dataset, or pd.DataFrame).
            group_dim (str): The dimension or column to group by for the Y-axis.
            x (str, optional): The variable/column to plot distributions of.
                Required if data is a Dataset or DataFrame with multiple variables.
            x_range (tuple[float, float], optional): Tuple (min, max) for the x-axis limits.
                If None, auto-calculated.
            scale_factor (float): Height scaling of the curves. Defaults to 1.0.
            overlap (float): Vertical spacing between curves. Higher values mean more overlap.
                Defaults to 0.5.
            cmap (str): Colormap name for coloring curves. Defaults to 'RdBu_r'.
            title (str, optional): Plot title.
            **kwargs: Additional keyword arguments for BasePlot (figure/axes creation).
        """
        super().__init__(**kwargs)
        self.data = normalize_data(data)
        self.group_dim = group_dim
        self.x = x
        self.x_range = x_range
        self.scale_factor = scale_factor
        self.overlap = overlap
        self.cmap_name = cmap
        self.title = title

    def plot(self, **kwargs: Any) -> matplotlib.axes.Axes:
        """
        Generate the ridgeline plot.

        Args:
            **kwargs: Additional keyword arguments for formatting.

        Returns:
            matplotlib.axes.Axes: The axes object containing the plot.
        """
        # 1. Prepare Data and Groups
        if isinstance(self.data, xr.DataArray):
            da = self.data
            da_sorted = da.sortby(self.group_dim, ascending=False)
            groups = da_sorted[self.group_dim].values
            all_values = da.values.flatten()
            data_name = da.name if da.name else "Value"
        elif isinstance(self.data, xr.Dataset):
            if self.x is None:
                self.x = list(self.data.data_vars)[0]
            da = self.data[self.x]
            da_sorted = da.sortby(self.group_dim, ascending=False)
            groups = da_sorted[self.group_dim].values
            all_values = da.values.flatten()
            data_name = da.name if da.name else self.x
        else:
            # Pandas DataFrame
            df = self.data
            if self.x is None:
                # Try to find a numeric column that is not group_dim
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                cols_to_use = [c for c in numeric_cols if c != self.group_dim]
                if not cols_to_use:
                    raise ValueError("No numeric columns found in DataFrame to plot.")
                self.x = cols_to_use[0]

            df_sorted = df.sort_values(self.group_dim, ascending=False)
            groups = df_sorted[self.group_dim].unique()
            all_values = df[self.x].values
            data_name = self.x

        all_values = all_values[~np.isnan(all_values)]
        if len(all_values) == 0:
            raise ValueError("No valid data points found to plot.")

        # Setup X-axis grid for density calculation
        if self.x_range is None:
            xmin, xmax = np.nanmin(all_values), np.nanmax(all_values)
            pad = (xmax - xmin) * 0.1
            x_grid = np.linspace(xmin - pad, xmax + pad, 200)
        else:
            x_grid = np.linspace(self.x_range[0], self.x_range[1], 200)

        # Setup Colors
        cmap, norm = get_linear_scale(all_values, cmap=self.cmap_name)

        # 2. Iterate and Plot
        for i, val in enumerate(groups):
            if isinstance(self.data, (xr.DataArray, xr.Dataset)):
                # Handle DataArray/Dataset slice
                data_slice = da_sorted.sel({self.group_dim: val}).values.flatten()
            else:
                # Handle DataFrame slice
                data_slice = df_sorted[df_sorted[self.group_dim] == val][
                    self.x
                ].values.flatten()

            data_slice = data_slice[~np.isnan(data_slice)]

            if len(data_slice) < 2:
                continue

            try:
                kde = gaussian_kde(data_slice)
                y_density = kde(x_grid)
            except (np.linalg.LinAlgError, ValueError):
                continue

            # Scale density and calculate vertical baseline
            y_density_scaled = y_density * self.scale_factor
            baseline = -i * self.overlap
            y_final = baseline + y_density_scaled

            # Color based on the mean of this slice
            slice_mean = np.mean(data_slice)
            color = cmap(norm(slice_mean))

            # Plot filling
            self.ax.fill_between(
                x_grid,
                baseline,
                y_final,
                facecolor=color,
                edgecolor="white",
                linewidth=0.5,
                alpha=0.9,
                zorder=len(groups) - i,
            )

            # Add Label for every 5th group to avoid clutter
            if i % 5 == 0:
                label_text = (
                    f"{val:.1f}"
                    if isinstance(val, (int, float, np.number))
                    else str(val)
                )
                self.ax.text(
                    x_grid[0],
                    baseline + (self.scale_factor * 0.1),
                    label_text,
                    fontsize=9,
                    color="#555",
                    verticalalignment="center",
                    ha="right",
                )

        # 3. Final Formatting
        self.ax.set_yticks([])
        self.ax.set_xlabel(data_name)
        if self.title:
            self.ax.set_title(self.title, pad=20)

        # Add a vertical zero line if the range crosses zero
        if x_grid.min() < 0 and x_grid.max() > 0:
            self.ax.axvline(0, color="black", alpha=0.3, linestyle="--", linewidth=1)

        # Remove unnecessary spines
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.spines["left"].set_visible(False)

        return self.ax
