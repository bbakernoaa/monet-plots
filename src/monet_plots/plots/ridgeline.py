# src/monet_plots/plots/ridgeline.py
"""Ridgeline (joyplot) plot implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple

import numpy as np
import xarray as xr
from scipy.stats import gaussian_kde

from ..colorbars import get_linear_scale
from ..plot_utils import _update_history, compute, normalize_data
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
        data: Any | None = None,
        group_dim: str | None = None,
        x: str | None = None,
        *,
        x_range: Tuple[float, float] | None = None,
        scale_factor: float = 1.0,
        overlap: float = 0.5,
        cmap: str = "viridis",
        title: str | None = None,
        bw_method: Any | None = None,
        alpha: float = 0.8,
        quantiles: list[float] | None = None,
        fig: Any | None = None,
        ax: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the ridgeline plot with data and visualization settings.

        Parameters
        ----------
        data : Any, optional
            The data to plot (xr.DataArray, xr.Dataset, or pd.DataFrame).
        group_dim : str, optional
            The dimension or column name to group by for the Y-axis.
        x : str, optional
            The variable or column name to plot distributions of.
            Required if data is a Dataset or DataFrame with multiple variables.
        x_range : tuple[float, float], optional
            Tuple (min, max) for the x-axis limits. If None, auto-calculated.
        scale_factor : float, optional
            Height scaling of the distribution curves, by default 1.0.
        overlap : float, optional
            Vertical spacing between curves. Higher values mean more overlap, by default 0.5.
        cmap : str, optional
            Colormap name for coloring curves, by default "viridis".
        title : str, optional
            Plot title, by default None.
        bw_method : Any, optional
            KDE bandwidth method (passed to `scipy.stats.gaussian_kde`), by default None.
        alpha : float, optional
            Transparency of the ridges, by default 0.8.
        quantiles : list[float], optional
            List of quantiles to display (e.g., [0.5]), by default None.
        fig : Any, optional
            Matplotlib figure instance, by default None.
        ax : Any, optional
            Matplotlib axes instance, by default None.
        **kwargs : Any
            Additional keyword arguments for BasePlot (figure/axes creation).

        Returns
        -------
        None

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from monet_plots.plots import RidgelinePlot
        >>> df = pd.DataFrame({'group': ['A']*50 + ['B']*50, 'val': np.random.randn(100)})
        >>> plot = RidgelinePlot(df, group_dim='group', x='val')
        """
        # Support legacy 'df' keyword if 'data' is not provided
        if data is None and "df" in kwargs:
            data = kwargs.pop("df")

        super().__init__(fig=fig, ax=ax, **kwargs)
        if self.ax is None:
            self.ax = self.fig.add_subplot(1, 1, 1)

        self.data = normalize_data(data)
        self.group_dim = group_dim
        self.x = x
        self.x_range = x_range
        self.scale_factor = scale_factor
        self.overlap = overlap
        self.cmap_name = cmap
        self.title = title
        self.bw_method = bw_method
        self.alpha = alpha
        self.quantiles = quantiles

        _update_history(
            self.data, f"Initialized RidgelinePlot for group_dim={group_dim}, x={x}"
        )

    def plot(
        self, gradient: bool = True, color_by_group: bool = False, **kwargs: Any
    ) -> matplotlib.axes.Axes:
        """
        Generate the ridgeline plot (Track A).

        Parameters
        ----------
        gradient : bool, optional
            If True, fill curves with a gradient based on x-values, by default True.
        color_by_group : bool, optional
            If True, color each ridge by its group category.
            Takes precedence over gradient, by default False.
        **kwargs : Any
            Additional formatting arguments.

        Returns
        -------
        matplotlib.axes.Axes
            The matplotlib axes object containing the plot.

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from monet_plots.plots import RidgelinePlot
        >>> df = pd.DataFrame({'group': ['A']*50 + ['B']*50, 'val': np.random.randn(100)})
        >>> plot = RidgelinePlot(df, group_dim='group', x='val')
        >>> ax = plot.plot()
        """
        import matplotlib.pyplot as plt

        # 1. Prepare Data and Groups
        if isinstance(self.data, xr.DataArray):
            da = self.data
            da_sorted = da.sortby(self.group_dim, ascending=False)
            groups = compute(da_sorted[self.group_dim])
            if hasattr(groups, "values"):
                groups = groups.values
            data_name = str(da.name) if da.name else "Value"

            if self.x_range is None:
                vmin_raw, vmax_raw = compute(da.min(), da.max())
                vmin, vmax = float(vmin_raw), float(vmax_raw)
            else:
                vmin, vmax = self.x_range

            if np.isnan(vmin) or np.isnan(vmax):
                raise ValueError("No valid data points found to plot.")

        elif isinstance(self.data, xr.Dataset):
            if self.x is None:
                self.x = list(self.data.data_vars)[0]
            da = self.data[self.x]
            da_sorted = da.sortby(self.group_dim, ascending=False)
            groups = compute(da_sorted[self.group_dim])
            if hasattr(groups, "values"):
                groups = groups.values
            data_name = str(da.name) if da.name else self.x

            if self.x_range is None:
                vmin_raw, vmax_raw = compute(da.min(), da.max())
                vmin, vmax = float(vmin_raw), float(vmax_raw)
            else:
                vmin, vmax = self.x_range

            if np.isnan(vmin) or np.isnan(vmax):
                raise ValueError("No valid data points found to plot.")

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
            data_name = str(self.x)

            if self.x_range is None:
                vmin = float(df[self.x].min())
                vmax = float(df[self.x].max())
            else:
                vmin, vmax = self.x_range

            if np.isnan(vmin) or np.isnan(vmax):
                raise ValueError("No valid data points found to plot.")

        # Setup X-axis grid for density calculation
        if self.x_range is None:
            pad = (vmax - vmin) * 0.1
            x_grid = np.linspace(vmin - pad, vmax + pad, 200)
        else:
            x_grid = np.linspace(vmin, vmax, 200)

        # Setup Colors
        cmap, norm = get_linear_scale(None, cmap=self.cmap_name, vmin=vmin, vmax=vmax)

        # 2. Iterate and Plot
        for i, val in enumerate(groups):
            if isinstance(self.data, (xr.DataArray, xr.Dataset)):
                # Handle DataArray/Dataset slice
                slice_raw = compute(da_sorted.sel({self.group_dim: val}))
                if hasattr(slice_raw, "values"):
                    data_slice = slice_raw.values.ravel()
                else:
                    data_slice = np.asarray(slice_raw).ravel()
            else:
                # Handle DataFrame slice
                data_slice = df_sorted[df_sorted[self.group_dim] == val][
                    self.x
                ].values.flatten()

            data_slice = data_slice[~np.isnan(data_slice)]

            if len(data_slice) < 2:
                continue

            try:
                kde = gaussian_kde(data_slice, bw_method=self.bw_method)
                y_density = kde(x_grid)
            except (np.linalg.LinAlgError, ValueError):
                continue

            # Scale density and calculate vertical baseline
            y_density_scaled = y_density * self.scale_factor
            baseline = -i * self.overlap
            y_final = baseline + y_density_scaled

            # Plot filling
            if color_by_group:
                # Use qualitative cmap or indexed colors
                color = plt.get_cmap("tab10")(i % 10)
                self.ax.fill_between(
                    x_grid,
                    baseline,
                    y_final,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=0.5,
                    alpha=self.alpha,
                    zorder=len(groups) - i,
                )
                self.ax.plot(
                    x_grid,
                    y_final,
                    color="black",
                    linewidth=0.5,
                    zorder=len(groups) - i + 0.1,
                )
            elif gradient:
                # Plot in segments to create a gradient effect
                for j in range(len(x_grid) - 1):
                    self.ax.fill_between(
                        x_grid[j : j + 2],
                        baseline,
                        y_final[j : j + 2],
                        facecolor=cmap(norm(x_grid[j])),
                        edgecolor="none",
                        alpha=self.alpha,
                        zorder=len(groups) - i,
                    )
                # Add a clean top outline
                self.ax.plot(
                    x_grid,
                    y_final,
                    color="black",
                    linewidth=0.5,
                    zorder=len(groups) - i + 0.1,
                )
            else:
                # Single color based on the mean of this slice
                slice_mean = np.mean(data_slice)
                color = cmap(norm(slice_mean))
                self.ax.fill_between(
                    x_grid,
                    baseline,
                    y_final,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=0.5,
                    alpha=self.alpha,
                    zorder=len(groups) - i,
                )
                self.ax.plot(
                    x_grid,
                    y_final,
                    color="black",
                    linewidth=0.5,
                    zorder=len(groups) - i + 0.1,
                )

            # 3. Add Quantiles
            if self.quantiles is not None:
                q_values = np.quantile(data_slice, self.quantiles)
                q_densities = kde(q_values) * self.scale_factor
                for qv, qd in zip(q_values, q_densities):
                    self.ax.vlines(
                        qv,
                        baseline,
                        baseline + qd,
                        color="black",
                        linestyle="--",
                        linewidth=0.8,
                        zorder=len(groups) - i + 0.2,
                    )

        # 3. Final Formatting
        self.ax.set_yticks([-i * self.overlap for i in range(len(groups))])
        self.ax.set_yticklabels(groups)
        self.ax.set_xlabel(data_name)
        if self.title:
            self.ax.set_title(self.title, pad=20)

        # Add Colorbar matching the x-axis scale if not coloring by group
        if not color_by_group:
            mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            self.add_colorbar(mappable, label=data_name)

        # Add vertical gridlines as seen in the reference
        self.ax.xaxis.grid(True, linestyle="-", alpha=0.3)

        # Add a vertical zero line if the range crosses zero
        if x_grid.min() < 0 and x_grid.max() > 0:
            self.ax.axvline(0, color="black", alpha=0.3, linestyle="--", linewidth=1)

        # Remove unnecessary spines
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.spines["left"].set_visible(False)

        if isinstance(self.data, (xr.DataArray, xr.Dataset)):
            _update_history(self.data, f"Created ridgeline plot for {data_name}")

        return self.ax

    def hvplot(self, **kwargs: Any) -> Any:
        """
        Generate an interactive ridgeline KDE plot using hvPlot (Track B).

        Parameters
        ----------
        **kwargs : Any
            Keyword arguments passed to `hvplot.kde`.

        Returns
        -------
        Any
            The interactive hvPlot object.

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from monet_plots.plots import RidgelinePlot
        >>> df = pd.DataFrame({'group': ['A']*50 + ['B']*50, 'val': np.random.randn(100)})
        >>> plot = RidgelinePlot(df, group_dim='group', x='val')
        >>> # interactive = plot.hvplot() # Requires hvplot installed
        """
        try:
            import hvplot.pandas  # noqa: F401
            import hvplot.xarray  # noqa: F401
        except ImportError:
            raise ImportError(
                "hvplot is required for interactive plotting. Install it with 'pip install hvplot'."
            )

        plot_kwargs = {}
        if self.x:
            plot_kwargs["y"] = self.x
        if self.group_dim:
            plot_kwargs["by"] = self.group_dim
        if self.title:
            plot_kwargs["title"] = self.title

        res = self.data.hvplot.kde(**{**plot_kwargs, **kwargs})

        _update_history(
            self.data,
            f"Generated interactive hvplot Ridgeline for group_dim={self.group_dim}",
        )
        return res
