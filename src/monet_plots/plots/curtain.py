# src/monet_plots/plots/curtain.py
"""Vertical curtain plot for cross-sectional data."""

from __future__ import annotations
from typing import Any, Optional

import matplotlib.pyplot as plt
import xarray as xr

from ..plot_utils import get_plot_kwargs, _update_history, normalize_data
from .base import BasePlot


class CurtainPlot(BasePlot):
    """Vertical curtain plot for cross-sectional data.

    This plot shows a 2D variable (e.g., concentration) as a function of
    one horizontal dimension (time or distance) and one vertical dimension
    (altitude or pressure). It also supports overlaying observation data.
    """

    def __init__(
        self,
        data: Any,
        *,
        x: Optional[str] = None,
        y: Optional[str] = None,
        obs_data: Optional[Any] = None,
        obs_var: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize Curtain Plot.

        Args:
            data: Input gridded data. Should be a 2D xarray.DataArray or similar.
            x: Name of the x-axis dimension/coordinate (e.g., 'time').
            y: Name of the y-axis dimension/coordinate (e.g., 'level').
            obs_data: Optional observation data to overlay.
            obs_var: Variable name in obs_data to plot.
            **kwargs: Arguments passed to BasePlot.
        """
        super().__init__(**kwargs)
        self.data = data
        self.x = x
        self.y = y
        self.obs_data = normalize_data(obs_data) if obs_data is not None else None
        self.obs_var = obs_var

        _update_history(self.data, "Initialized monet-plots.CurtainPlot")

    def plot(
        self,
        kind: str = "contourf",
        colorbar: bool = True,
        overlay_obs: bool = True,
        two_subplot: bool = False,
        cbar_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """
        Generate the curtain plot.

        Args:
            kind: Type of plot ('pcolormesh' or 'contourf').
            colorbar: Whether to add a colorbar.
            overlay_obs: Whether to overlay observation scatter if obs_data is provided.
            two_subplot: If True, creates two subplots (1: model+overlay, 2: obs only).
            cbar_kwargs: Arguments for colorbar creation.
            **kwargs: Additional arguments for the plotting function.
        """
        plot_kwargs = get_plot_kwargs(**kwargs)

        # Ensure we have a DataArray for gridded data
        if not isinstance(self.data, xr.DataArray):
            if hasattr(self.data, "to_array"):
                da = self.data.to_array()
            else:
                da = xr.DataArray(self.data)
        else:
            da = self.data

        if da.ndim != 2:
            raise ValueError(f"CurtainPlot requires 2D data, got {da.ndim}D.")

        # Determine x and y if not provided
        if self.x is None:
            self.x = da.dims[1]
        if self.y is None:
            self.y = da.dims[0]

        if two_subplot:
            # Re-initialize fig/ax for two subplots
            plt.close(self.fig)
            self.fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
            self.ax = ax1
        else:
            if self.ax is None:
                self.ax = self.fig.add_subplot(1, 1, 1)
            ax1 = self.ax
            ax2 = None

        # Plot gridded data on ax1
        if kind == "pcolormesh":
            mappable = ax1.pcolormesh(
                da[self.x], da[self.y], da, shading="auto", **plot_kwargs
            )
        elif kind == "contourf":
            mappable = ax1.contourf(da[self.x], da[self.y], da, **plot_kwargs)
        else:
            raise ValueError("kind must be 'pcolormesh' or 'contourf'")

        ax1.set_ylabel(self.y)
        if ax2 is None:
            ax1.set_xlabel(self.x)

        # Overlay observations
        if self.obs_data is not None and self.obs_var is not None:
            # Handle overlay on ax1
            if overlay_obs:
                ax1.scatter(
                    self.obs_data[self.x],
                    self.obs_data[self.y],
                    c=self.obs_data[self.obs_var],
                    norm=mappable.norm,
                    cmap=mappable.cmap,
                    edgecolors="k",
                    alpha=0.7,
                    label="Obs Overlay",
                )

            # Handle second subplot for observations only
            if ax2 is not None:
                ax2.scatter(
                    self.obs_data[self.x],
                    self.obs_data[self.y],
                    c=self.obs_data[self.obs_var],
                    norm=mappable.norm,
                    cmap=mappable.cmap,
                    edgecolors="k",
                    alpha=0.7,
                )
                ax2.set_ylabel(self.y)
                ax2.set_xlabel(self.x)
                ax2.set_title("Observations Only")

        # Invert y-axis if pressure
        if "pressure" in str(self.y).lower():
            ax1.invert_yaxis()
            if ax2 is not None:
                ax2.invert_yaxis()

        if colorbar:
            cb_kwargs = cbar_kwargs or {}
            if two_subplot:
                # Shared colorbar for both subplots
                self.fig.colorbar(
                    mappable,
                    ax=[ax1, ax2],
                    orientation=cb_kwargs.pop("orientation", "horizontal"),
                    pad=cb_kwargs.pop("pad", 0.1),
                    aspect=cb_kwargs.pop("aspect", 50),
                    **cb_kwargs,
                )
            else:
                self.add_colorbar(mappable, ax=ax1, **cb_kwargs)

        if two_subplot:
            ax1.set_title("Model with Obs Overlay")
            self.fig.autofmt_xdate()

        return self.ax

    def hvplot(self, **kwargs: Any) -> Any:
        """Generate an interactive curtain plot using hvPlot (Track B).

        Parameters
        ----------
        **kwargs : Any
            Keyword arguments passed to `hvplot`.
            Common options include `cmap`, `title`, and `alpha`.

        Returns
        -------
        holoviews.core.layout.Layout
            The interactive hvPlot object.
        """
        try:
            import hvplot.xarray  # noqa: F401
        except ImportError:
            raise ImportError(
                "hvplot is required for interactive plotting. Install it with 'pip install hvplot'."
            )

        # Track B defaults for gridded data
        da = self.data if isinstance(self.data, xr.DataArray) else self.data.to_array()

        plot_gridded = da.hvplot(
            x=self.x,
            y=self.y,
            rasterize=True,
            title="Interactive Curtain Plot",
            **kwargs,
        )

        if self.obs_data is not None and self.obs_var is not None:
            plot_obs = self.obs_data.hvplot.scatter(
                x=self.x,
                y=self.y,
                c=self.obs_var,
                cmap=kwargs.get("cmap", "viridis"),
                **kwargs,
            )
            return plot_gridded * plot_obs

        return plot_gridded
