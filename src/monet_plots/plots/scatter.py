# src/monet_plots/plots/scatter.py
"""Scatter plot with regression line supporting lazy evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, Union

import cartopy.crs as ccrs
import numpy as np
import xarray as xr

from ..plot_utils import normalize_data
from .base import BasePlot

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure


class ScatterPlot(BasePlot):
    """
    Create a scatter plot with a regression line (Unified API).

    This plot shows the relationship between two variables and includes a
    linear regression model fit. It supports lazy evaluation for large
    Xarray/Dask datasets by delaying computation until the plot call.

    Parameters
    ----------
    data : Any
        The input data for the plot.
    var1 : str
        The name of the variable for the x-axis.
    var2 : str or list of str
        The name(s) of the variable(s) for the y-axis.
    c : Optional[str]
        The name of the variable used for colorizing points.
    colorbar : bool
        Whether to add a colorbar to the plot.
    title : Optional[str]
        The title for the plot.
    fig : matplotlib.figure.Figure, optional
        An existing Figure object.
    ax : matplotlib.axes.Axes, optional
        An existing Axes object.
    df : Any, optional
        Deprecated alias for ``data``.
    x : str, optional
        Legacy alias for var1.
    y : str or list, optional
        Legacy alias for var2.
    **kwargs : Any
        Additional keyword arguments passed to BasePlot.
    """

    def __init__(
        self,
        data: Any = None,
        var1: Optional[str] = None,
        var2: Optional[Union[str, List[str]]] = None,
        c: Optional[str] = None,
        colorbar: bool = False,
        title: Optional[str] = None,
        fig: Optional[matplotlib.figure.Figure] = None,
        ax: Optional[matplotlib.axes.Axes] = None,
        df: Any = None,  # Backward compatibility alias
        x: Optional[str] = None,  # legacy alias
        y: Optional[Union[str, List[str]]] = None,  # legacy alias
        **kwargs: Any,
    ) -> None:
        super().__init__(fig=fig, ax=ax, **kwargs)
        if self.ax is None:
            self.ax = self.fig.add_subplot(1, 1, 1)

        if df is not None and data is None:
            data = df
        self.data = normalize_data(data)
        self.var1 = var1 or x
        if var2 is not None:
            self.var2 = [var2] if isinstance(var2, str) else var2
        elif y is not None:
            self.var2 = [y] if isinstance(y, str) else y
        else:
            self.var2 = []
        self.c = c
        self.colorbar = colorbar
        self.title = title

        if not self.var1 or not self.var2:
            raise ValueError("Parameters 'var1' and 'var2' must be provided.")

        # Update history for provenance if Xarray
        if isinstance(self.data, (xr.DataArray, xr.Dataset)):
            history = self.data.attrs.get("history", "")
            self.data.attrs["history"] = f"Initialized ScatterPlot; {history}"

    def _get_regression_line(
        self, x_val: np.ndarray, y_val: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate regression line points using only endpoints.

        Parameters
        ----------
        x_val : np.ndarray
            The concrete x-axis data.
        y_val : np.ndarray
            The concrete y-axis data.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The x and y values for the regression line endpoints.
        """
        mask = ~np.isnan(x_val) & ~np.isnan(y_val)
        if not np.any(mask):
            return np.array([np.nan, np.nan]), np.array([np.nan, np.nan])

        m, b = np.polyfit(x_val[mask], y_val[mask], 1)
        x_min, x_max = np.nanmin(x_val), np.nanmax(x_val)
        x_reg = np.array([x_min, x_max])
        y_reg = m * x_reg + b
        return x_reg, y_reg

    def plot(
        self,
        scatter_kws: Optional[dict[str, Any]] = None,
        line_kws: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> matplotlib.axes.Axes:
        """Generate a static publication-quality scatter plot (Track A).

        Parameters
        ----------
        scatter_kws : dict, optional
            Additional keyword arguments for `ax.scatter`.
        line_kws : dict, optional
            Additional keyword arguments for the regression `ax.plot`.
        **kwargs : Any
            Secondary way to pass keyword arguments to `ax.scatter`.
            Merged with `scatter_kws`.

        Returns
        -------
        matplotlib.axes.Axes
            The axes object with the scatter plot.

        Notes
        -----
        For massive datasets (> RAM), consider using Track B (Exploration)
        tools like `hvplot` with `rasterize=True`.
        """
        from ..plot_utils import get_plot_kwargs

        # Combine scatter_kws and kwargs
        s_kws = scatter_kws.copy() if scatter_kws is not None else {}
        s_kws.update(kwargs)

        l_kws = line_kws.copy() if line_kws is not None else {}

        # Aero Protocol Requirement: Mandatory transform for GeoAxes
        is_geo = hasattr(self.ax, "projection")
        if is_geo:
            s_kws.setdefault("transform", ccrs.PlateCarree())
            l_kws.setdefault("transform", ccrs.PlateCarree())

        transform = s_kws.get("transform")

        # Performance: Compute required variables once to avoid double work

        cols = [self.var1] + self.var2
        if self.c:
            cols.append(self.c)

        if hasattr(self.data, "compute"):
            # Sub-selection before compute to minimize data transfer
            subset = self.data[cols]
            concrete_data = subset.compute()
        else:
            concrete_data = self.data

        x_plot = concrete_data[self.var1].values.flatten()

        for y_col in self.var2:
            y_plot = concrete_data[y_col].values.flatten()

            if self.c is not None:
                c_plot = concrete_data[self.c].values.flatten()

                final_s_kwargs = get_plot_kwargs(c=c_plot, **s_kws)
                mappable = self.ax.scatter(x_plot, y_plot, **final_s_kwargs)

                if self.colorbar:
                    self.add_colorbar(mappable)
            else:
                final_s_kwargs = s_kws.copy()
                final_s_kwargs.setdefault("label", y_col)
                self.ax.scatter(x_plot, y_plot, **final_s_kwargs)

            # Add regression line using endpoints
            x_reg, y_reg = self._get_regression_line(x_plot, y_plot)

            final_l_kwargs = {
                "color": "#333333",
                "linestyle": "--",
                "linewidth": 1.5,
                "label": "Fit" if (self.c is None and len(self.y) == 1) else None,
            }
            final_l_kwargs.update(l_kws)
            if transform:
                final_l_kwargs.setdefault("transform", transform)

            self.ax.plot(x_reg, y_reg, **final_l_kwargs)


        if len(self.var2) > 1 and self.c is None:
            self.ax.legend()

        if self.title:
            self.ax.set_title(self.title)
        else:
            self.ax.set_title(f"Scatter: {self.var1} vs {', '.join(self.var2)}")

        self.ax.set_xlabel(self.var1)
        self.ax.set_ylabel(", ".join(self.var2) if len(self.var2) > 1 else self.var2[0])

        # Update history for provenance
        if isinstance(self.data, (xr.DataArray, xr.Dataset)):
            history = self.data.attrs.get("history", "")
            self.data.attrs["history"] = f"Generated ScatterPlot; {history}"

        return self.ax

    def hvplot(self, **kwargs: Any) -> Any:
        """Generate an interactive scatter plot using hvPlot (Track B).

        Parameters
        ----------
        **kwargs : Any
            Keyword arguments passed to `hvplot.scatter`.
            Common options include `cmap`, `title`, and `alpha`.
            `rasterize=True` is used by default for high performance.

        Returns
        -------
        holoviews.core.layout.Layout
            The interactive hvPlot object.
        """
        try:
            import hvplot.pandas  # noqa: F401
            import hvplot.xarray  # noqa: F401
        except ImportError:
            raise ImportError(
                "hvplot is required for interactive plotting. Install it with 'pip install hvplot'."
            )

        # Track B defaults
        plot_kwargs = {
            "x": self.x,
            "y": self.y[0] if len(self.y) == 1 else self.y,
            "rasterize": True,
        }
        if self.c:
            plot_kwargs["c"] = self.c

        plot_kwargs.update(kwargs)

        return self.data.hvplot.scatter(**plot_kwargs)
