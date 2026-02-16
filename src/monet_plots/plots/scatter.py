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
    """Create a scatter plot with a regression line.

    This plot shows the relationship between two variables and includes a
    linear regression model fit. It supports lazy evaluation for large
    Xarray/Dask datasets by delaying computation until the plot call.

    Attributes
    ----------
    data : Union[xr.Dataset, xr.DataArray, pd.DataFrame]
        The input data for the plot.
    x : str
        The name of the variable for the x-axis.
    y : List[str]
        The names of the variables for the y-axis.
    c : Optional[str]
        The name of the variable used for colorizing points.
    colorbar : bool
        Whether to add a colorbar to the plot.
    title : Optional[str]
        The title for the plot.
    """

    def __init__(
        self,
        data: Any = None,
        x: Optional[str] = None,
        y: Optional[Union[str, List[str]]] = None,
        c: Optional[str] = None,
        colorbar: bool = False,
        title: Optional[str] = None,
        fig: Optional[matplotlib.figure.Figure] = None,
        ax: Optional[matplotlib.axes.Axes] = None,
        df: Any = None,  # Backward compatibility alias
        **kwargs: Any,
    ) -> None:
        """
        Initialize the scatter plot.

        Parameters
        ----------
        data : Any, optional
            Input data. Can be a pandas DataFrame, xarray DataArray,
            xarray Dataset, or numpy ndarray.
        x : str, optional
            Variable name for the x-axis.
        y : Union[str, List[str]], optional
            Variable name(s) for the y-axis.
        c : str, optional
            Variable name for colorizing the points, by default None.
        colorbar : bool, optional
            Whether to add a colorbar, by default False.
        title : str, optional
            Title for the plot, by default None.
        fig : matplotlib.figure.Figure, optional
            An existing Figure object.
        ax : matplotlib.axes.Axes, optional
            An existing Axes object.
        df : Any, optional
            Alias for `data` for backward compatibility.
        **kwargs : Any
            Additional keyword arguments passed to BasePlot.
        """
        super().__init__(fig=fig, ax=ax, **kwargs)
        if self.ax is None:
            self.ax = self.fig.add_subplot(1, 1, 1)

        self.data = normalize_data(data if data is not None else df)
        self.x = x
        self.y = [y] if isinstance(y, str) else (y if y is not None else [])
        self.c = c
        self.colorbar = colorbar
        self.title = title

        if not self.x or not self.y:
            raise ValueError("Parameters 'x' and 'y' must be provided.")

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
        """Generate the scatter plot.

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
        cols = [self.x] + self.y
        if self.c:
            cols.append(self.c)

        if hasattr(self.data, "compute"):
            # Sub-selection before compute to minimize data transfer
            subset = self.data[cols]
            concrete_data = subset.compute()
        else:
            concrete_data = self.data

        x_plot = concrete_data[self.x].values.flatten()

        for y_col in self.y:
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
                "color": "red",
                "linestyle": "--",
                "label": "Fit" if (self.c is None and len(self.y) == 1) else None,
            }
            final_l_kwargs.update(l_kws)
            if transform:
                final_l_kwargs.setdefault("transform", transform)

            self.ax.plot(x_reg, y_reg, **final_l_kwargs)

        if len(self.y) > 1 and self.c is None:
            self.ax.legend()

        if self.title:
            self.ax.set_title(self.title)
        else:
            self.ax.set_title(f"Scatter: {self.x} vs {', '.join(self.y)}")

        self.ax.set_xlabel(self.x)
        self.ax.set_ylabel(", ".join(self.y) if len(self.y) > 1 else self.y[0])

        # Update history for provenance
        if isinstance(self.data, (xr.DataArray, xr.Dataset)):
            history = self.data.attrs.get("history", "")
            self.data.attrs["history"] = f"Generated ScatterPlot; {history}"

        return self.ax

    def hvplot(self, **kwargs: Any) -> Any:
        """
        Generate an interactive scatter plot using hvPlot.

        This method follows Track B of the Aero Protocol, providing an
        interactive visualization suitable for exploration.

        Parameters
        ----------
        **kwargs : Any
            Additional keyword arguments passed to `hvplot.scatter`.
            Common options include `cmap`, `rasterize`, and `title`.

        Returns
        -------
        holoviews.Element
            The interactive HoloViews object.
        """
        import hvplot.pandas  # noqa: F401
        import hvplot.xarray  # noqa: F401

        if self.data is None:
            raise ValueError("Data must be provided during initialization for hvplot()")

        # Default settings
        kwargs.setdefault(
            "title",
            self.title if self.title else f"Scatter: {self.x} vs {', '.join(self.y)}",
        )
        if self.c:
            kwargs.setdefault("c", self.c)

        # Performance: Use rasterize=True for large datasets (>100MB)
        if hasattr(self.data, "nbytes") and self.data.nbytes > 1e8:
            kwargs.setdefault("rasterize", True)

        plot = self.data.hvplot.scatter(x=self.x, y=self.y, **kwargs)

        # Provenance
        from ..verification_metrics import _update_history

        _update_history(self.data, "Generated interactive ScatterPlot")

        return plot
