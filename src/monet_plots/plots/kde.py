# src/monet_plots/plots/kde.py
"""Kernel Density Estimate (KDE) plot implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import seaborn as sns
import xarray as xr

from ..plot_utils import _update_history, normalize_data
from .base import BasePlot

if TYPE_CHECKING:
    import matplotlib.axes


class KDEPlot(BasePlot):
    """
    Create a kernel density estimate plot.

    This plot shows the distribution of a single variable or the joint
    distribution of two variables using kernel density estimation.

    Attributes
    ----------
    data : Any
        The normalized input data.
    x : str
        Column or variable name for the x-axis.
    y : str | None
        Column or variable name for the y-axis (for bivariate plots).
    title : str | None
        The plot title.
    label : str | None
        The plot label for legends.
    """

    def __init__(
        self,
        data: Any | None = None,
        x: str | None = None,
        y: str | None = None,
        *,
        title: str | None = None,
        label: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the plot with data and plot settings.

        Parameters
        ----------
        data : Any, optional
            The data to plot. Can be a pandas DataFrame, xarray DataArray,
            xarray Dataset, or numpy ndarray, by default None.
        x : str, optional
            Column/variable name for the x-axis, by default None.
        y : str, optional
            Column/variable name for the y-axis (for bivariate plots).
            If None, a univariate KDE is created, by default None.
        title : str, optional
            Title for the plot, by default None.
        label : str, optional
            Label for the plot (used in legends), by default None.
        **kwargs : Any
            Additional keyword arguments for BasePlot (figure/axes creation).

        Returns
        -------
        None

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from monet_plots.plots import KDEPlot
        >>> df = pd.DataFrame({'val': np.random.randn(100)})
        >>> plot = KDEPlot(df, x='val')
        """
        # Support legacy 'df' keyword if 'data' is not provided
        if data is None and "df" in kwargs:
            data = kwargs.pop("df")

        super().__init__(**kwargs)
        self.data = normalize_data(data)
        self.x = x
        self.y = y
        self.title = title
        self.label = label

        if self.x is None:
            raise ValueError("Parameter 'x' must be provided.")

        _update_history(self.data, f"Initialized KDEPlot for x={x}, y={y}")

    def plot(self, **kwargs: Any) -> matplotlib.axes.Axes:
        """
        Generate the KDE plot using Seaborn (Track A).

        Parameters
        ----------
        **kwargs : Any
            Additional keyword arguments passed to `sns.kdeplot`.

        Returns
        -------
        matplotlib.axes.Axes
            The axes object containing the plot.

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from monet_plots.plots import KDEPlot
        >>> df = pd.DataFrame({'val': np.random.randn(100)})
        >>> plot = KDEPlot(df, x='val', title='Univariate KDE')
        >>> ax = plot.plot(fill=True)
        """
        # Seaborn doesn't support xr.DataArray as a 'data' source directly if it's 1D,
        # it expects a DataFrame or Mapping. Converting to Dataset or DataFrame.
        data_to_plot = self.data
        if isinstance(data_to_plot, xr.DataArray):
            data_to_plot = data_to_plot.to_dataset()

        with sns.axes_style("ticks"):
            self.ax = sns.kdeplot(
                data=data_to_plot,
                x=self.x,
                y=self.y,
                ax=self.ax,
                label=self.label,
                **kwargs,
            )
            if self.title:
                self.ax.set_title(self.title)
            sns.despine()

        _update_history(self.data, f"Plotted KDE for x={self.x}, y={self.y}")
        return self.ax

    def hvplot(self, **kwargs: Any) -> Any:
        """
        Generate an interactive KDE plot using hvPlot (Track B).

        Parameters
        ----------
        **kwargs : Any
            Keyword arguments passed to `hvplot.kde`.
            For bivariate plots, these are passed to `hvplot.kde` or `hvplot.hexbin`.
            `rasterize=True` is used by default for bivariate high performance.

        Returns
        -------
        Any
            The interactive hvPlot object.

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from monet_plots.plots import KDEPlot
        >>> df = pd.DataFrame({'val': np.random.randn(100)})
        >>> plot = KDEPlot(df, x='val')
        >>> # interactive = plot.hvplot() # Requires hvplot installed
        """
        try:
            import hvplot.pandas  # noqa: F401
            import hvplot.xarray  # noqa: F401
        except ImportError:
            raise ImportError(
                "hvplot is required for interactive plotting. Install it with 'pip install hvplot'."
            )

        plot_kwargs = {"x": self.x}
        if self.y:
            plot_kwargs["y"] = self.y
            plot_kwargs.setdefault("rasterize", True)
        if self.title:
            plot_kwargs["title"] = self.title
        if self.label:
            plot_kwargs["label"] = self.label

        res = self.data.hvplot.kde(**{**plot_kwargs, **kwargs})

        _update_history(self.data, f"Generated interactive hvplot KDE for x={self.x}")
        return res
