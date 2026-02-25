from __future__ import annotations

import typing as t

import xarray as xr
from matplotlib import pyplot as plt

from ..plot_utils import _update_history, normalize_data
from .base import BasePlot


class Meteogram(BasePlot):
    """Meteogram plot for time series data.

    A meteogram is a time series plot of one or more meteorological variables
    at a specific location. This implementation supports multiple subplots,
    one for each variable, sharing a common x-axis (time).
    """

    def __init__(
        self,
        data: t.Any | None = None,
        variables: list[str] | None = None,
        x: str = "time",
        df: t.Any | None = None,
        **kwargs: t.Any,
    ) -> None:
        """
        Initialize the Meteogram plot.

        Parameters
        ----------
        data : pd.DataFrame, xr.Dataset, or xr.DataArray, optional
            The input data containing time series for the specified variables.
        variables : list of str, optional
            The list of variables to be plotted in individual subplots.
        x : str, optional
            The name of the time dimension or column, by default "time".
        df : pd.DataFrame, optional
            Legacy alias for `data` for backward compatibility.
        **kwargs : t.Any
            Additional keyword arguments passed to BasePlot.
        """
        if "fig" not in kwargs and "ax" not in kwargs:
            kwargs["fig"] = plt.figure()
        super().__init__(**kwargs)

        data = data if data is not None else df
        if data is None:
            raise ValueError("Input data must be provided as 'data' or 'df'.")

        if variables is None:
            raise ValueError("The 'variables' list must be provided.")

        self.data = normalize_data(data)
        self.variables = variables
        self.x = x

        # Update history for provenance
        _update_history(self.data, "Initialized Meteogram")

    def plot(self, **kwargs: t.Any) -> list[plt.Axes]:
        """
        Generate the static Meteogram plot using Matplotlib.

        Parameters
        ----------
        **kwargs : t.Any
            Keyword arguments passed to `matplotlib.pyplot.plot`.

        Returns
        -------
        list of matplotlib.axes.Axes
            The list of axes objects for each subplot.
        """
        if self.fig is None:
            self.fig = plt.figure()

        n_vars = len(self.variables)
        axes = []

        # Update history for provenance
        _update_history(self.data, f"Generated static plot for {self.variables}")

        for i, var in enumerate(self.variables):
            ax = self.fig.add_subplot(n_vars, 1, i + 1)
            axes.append(ax)

            if isinstance(self.data, (xr.Dataset, xr.DataArray)):
                # Xarray path
                da = self.data[var]
                da.plot(ax=ax, x=self.x, **kwargs)
            else:
                # Pandas path
                # Ensure x-axis is handled correctly if it's the index
                if self.x == "index" or self.x not in self.data.columns:
                    x_data = self.data.index
                else:
                    x_data = self.data[self.x]
                ax.plot(x_data, self.data[var], **kwargs)

            ax.set_ylabel(var)
            if i < n_vars - 1:
                plt.setp(ax.get_xticklabels(), visible=False)
                ax.set_xlabel("")

        self.ax = axes
        return axes

    def hvplot(self, **kwargs: t.Any) -> t.Any:
        """
        Generate an interactive Meteogram plot using hvPlot (Track B).

        Parameters
        ----------
        **kwargs : t.Any
            Keyword arguments passed to `hvplot()`.

        Returns
        -------
        holoviews.core.layout.Layout
            The interactive layout of plots.
        """
        import hvplot.pandas  # noqa: F401
        import hvplot.xarray  # noqa: F401

        # Update history for provenance
        _update_history(self.data, f"Generated interactive plot for {self.variables}")

        plots = []
        for var in self.variables:
            if isinstance(self.data, (xr.Dataset, xr.DataArray)):
                p = self.data[var].hvplot(x=self.x, **kwargs)
            else:
                p = self.data.hvplot(x=self.x, y=var, **kwargs)
            plots.append(p)

        import holoviews as hv

        return hv.Layout(plots).cols(1)
