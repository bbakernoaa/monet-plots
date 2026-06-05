from __future__ import annotations

import typing as t
from typing import Any, List, Optional

import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt

from ..plot_utils import _update_history, normalize_data
from .base import BasePlot

if t.TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure


class Meteogram(BasePlot):
    """Meteogram plot for time series data.

    A meteogram displays multiple variables as a function of time,
    usually in separate stacked subplots sharing a common x-axis.
    """

    def __init__(
        self,
        data: Optional[Any] = None,
        *,
        df: Optional[Any] = None,
        variables: List[str],
        x: str = "time",
        fig: Optional[matplotlib.figure.Figure] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the Meteogram.

        Parameters
        ----------
        data : Any, optional
            Input data. Can be a pandas DataFrame, xarray DataArray,
            xarray Dataset, or numpy ndarray.
        df : Any, optional
            Alias for `data` for backward compatibility.
        variables : list of str
            List of variables to plot in separate subplots.
        x : str, optional
            The time dimension/column name, by default "time".
        fig : matplotlib.figure.Figure, optional
            An existing Figure object.
        **kwargs : Any
            Additional keyword arguments passed to BasePlot.
        """
        # For Meteogram, we often want to create a new figure with multiple subplots.
        # BasePlot handles fig creation if not provided.
        super().__init__(fig=fig, **kwargs)

        # Support both 'data' and 'df' for backward compatibility
        input_data = data if data is not None else df
        if input_data is None:
            raise ValueError("Must provide data to Meteogram.")

        self.df = normalize_data(input_data)

        # Ensure xarray DataArray is converted to Dataset for consistent indexing
        if isinstance(self.df, xr.DataArray):
            if self.df.name is not None:
                self.df = self.df.to_dataset()
            else:
                # If no name, and only one variable is requested, try using it as name
                if len(variables) == 1:
                    self.df = self.df.to_dataset(name=variables[0])
                else:
                    self.df = self.df.to_dataset(name="data")

        self.variables = variables
        self.x = x

        # Track provenance for Xarray
        _update_history(self.df, "Initialized Meteogram")

    def plot(self, **kwargs: Any) -> List[matplotlib.axes.Axes]:
        """
        Generate the meteogram plot.

        Parameters
        ----------
        **kwargs : Any
            Keyword arguments passed to each variable's plot call.

        Returns
        -------
        list of matplotlib.axes.Axes
            The list of axes objects created for the meteogram.

        Examples
        --------
        >>> meteo = Meteogram(data=ds, variables=['temp', 'rh'])
        >>> axs = meteo.plot(color='blue')
        """
        n_vars = len(self.variables)

        # Clear existing axes if we are re-plotting on the same figure
        # (unless they were explicitly passed, but Meteogram usually manages its own layout)
        if self.fig.get_axes():
            for ax in self.fig.get_axes():
                ax.remove()

        axs = []
        for i, var in enumerate(self.variables):
            ax = self.fig.add_subplot(n_vars, 1, i + 1)
            axs.append(ax)

            if isinstance(self.df, xr.Dataset):
                # Xarray plotting
                self.df[var].plot(ax=ax, x=self.x, **kwargs)
                # Cleanup xarray auto-labels for intermediate plots
                if i < n_vars - 1:
                    ax.set_xlabel("")
            else:
                # Pandas plotting
                ax.plot(self.df[self.x], self.df[var], **kwargs)
                ax.set_ylabel(var)

            if i < n_vars - 1:
                plt.setp(ax.get_xticklabels(), visible=False)

        self.ax = axs

        # Update history for provenance
        _update_history(self.df, "Generated Meteogram Plot")

        return axs

    def hvplot(self, **kwargs: Any) -> Any:
        """
        Generate an interactive meteogram using hvPlot (Track B).

        Parameters
        ----------
        **kwargs : Any
            Keyword arguments passed to `hvplot`.

        Returns
        -------
        holoviews.core.Layout
            The interactive layout of subplots.

        Examples
        --------
        >>> meteo = Meteogram(data=ds, variables=['temp', 'rh'])
        >>> layout = meteo.hvplot(width=800, height=300)
        """
        try:
            import hvplot.pandas  # noqa: F401
            import hvplot.xarray  # noqa: F401
        except ImportError:
            raise ImportError(
                "hvplot is required for interactive plotting. Install it with 'pip install hvplot'."
            )

        plots = []
        for var in self.variables:
            p = self.df.hvplot(x=self.x, y=var, **kwargs)
            plots.append(p)

        import holoviews as hv

        return hv.Layout(plots).cols(1)
