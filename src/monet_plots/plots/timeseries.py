# src/monet_plots/plots/timeseries.py
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

from ..plot_utils import normalize_data
from .base import BasePlot

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure


class TimeSeriesPlot(BasePlot):
    """Create a timeseries plot with shaded error bounds.

    This function groups the data by time, plots the mean values, and adds
    shading for ±1 standard deviation around the mean.
    """

    def __init__(
        self,
        df: Any,
        x: str = "time",
        y: str = "obs",
        plotargs: dict = {},
        fillargs: dict = None,
        title: str = "",
        ylabel: Optional[str] = None,
        label: Optional[str] = None,
        *args,
        **kwargs,
    ):
        """
        Initialize the plot with data and plot settings.

        Args:
            df (pd.DataFrame, np.ndarray, xr.Dataset, xr.DataArray):
                DataFrame with the data to plot.
            x (str): Column name for the x-axis (time).
            y (str): Column name for the y-axis (values).
            plotargs (dict): Arguments for the plot.
            fillargs (dict): Arguments for fill_between.
            title (str): Title for the plot.
            ylabel (str, optional): Y-axis label.
            label (str, optional): Label for the plotted line.
            *args, **kwargs: Arguments passed to BasePlot.
        """
        super().__init__(*args, **kwargs)
        if self.ax is None:
            self.ax = self.fig.add_subplot(1, 1, 1)

        self.df = normalize_data(df)
        self.x = x
        self.y = y
        self.plotargs = plotargs
        self.fillargs = fillargs if fillargs is not None else {"alpha": 0.2}
        self.title = title
        self.ylabel = ylabel
        self.label = label

    def plot(self, **kwargs: Any) -> matplotlib.axes.Axes:
        """
        Generate the timeseries plot.

        Parameters
        ----------
        **kwargs : Any
            Overrides for plot settings (x, y, title, ylabel, label, etc.).

        Returns
        -------
        matplotlib.axes.Axes
            The matplotlib axes object containing the plot.

        Examples
        --------
        >>> plot = TimeSeriesPlot(df, x='time', y='obs')
        >>> ax = plot.plot(title='Observation Over Time')
        """
        # Update attributes from kwargs if provided
        for attr in ["x", "y", "title", "ylabel", "label"]:
            if attr in kwargs:
                setattr(self, attr, kwargs.pop(attr))

        import xarray as xr

        # Handle xarray objects differently from pandas DataFrames
        if isinstance(self.df, (xr.DataArray, xr.Dataset)):
            return self._plot_xarray(**kwargs)
        else:
            return self._plot_dataframe(**kwargs)

    def _plot_dataframe(self, **kwargs: Any) -> matplotlib.axes.Axes:
        """
        Generate the timeseries plot from pandas DataFrame.

        Parameters
        ----------
        **kwargs : Any
            Additional plotting arguments.

        Returns
        -------
        matplotlib.axes.Axes
            The matplotlib axes object.

        Examples
        --------
        >>> plot._plot_dataframe()
        """
        df = self.df.copy()
        df.index = df[self.x]
        # Keep only numeric columns for grouping, but make sure self.y is there
        df = df.reset_index(drop=True)
        # We need to preserve self.x for grouping if it's not the index
        m = self.df.groupby(self.x).mean(numeric_only=True)
        e = self.df.groupby(self.x).std(numeric_only=True)

        variable = self.y
        unit = "None"
        if "units" in self.df.columns:
            unit = str(self.df["units"].iloc[0])

        upper = m[self.y] + e[self.y]
        lower = m[self.y] - e[self.y]
        # lower.loc[lower < 0] = 0 # Not always desired for all variables
        lower_vals = lower.values
        upper_vals = upper.values

        if self.label is not None:
            plot_label = self.label
        else:
            plot_label = self.y

        m[self.y].plot(ax=self.ax, label=plot_label, **self.plotargs)
        self.ax.fill_between(m.index, lower_vals, upper_vals, **self.fillargs)

        if self.ylabel is None:
            self.ax.set_ylabel(f"{variable} ({unit})")
        else:
            self.ax.set_ylabel(self.ylabel)

        self.ax.set_xlabel(self.x)
        self.ax.legend()
        self.ax.set_title(self.title)
        self.fig.tight_layout()
        return self.ax

    def _plot_xarray(self, **kwargs: Any) -> matplotlib.axes.Axes:
        """
        Generate the timeseries plot from xarray DataArray or Dataset.

        Parameters
        ----------
        **kwargs : Any
            Additional plotting arguments.

        Returns
        -------
        matplotlib.axes.Axes
            The matplotlib axes object.

        Examples
        --------
        >>> plot._plot_xarray()
        """
        import xarray as xr

        # Ensure we have the right data structure
        if isinstance(self.df, xr.DataArray):
            data = (
                self.df.to_dataset(name=self.y)
                if self.df.name is None
                else self.df.to_dataset()
            )
            if self.df.name is not None:
                self.y = self.df.name
        else:
            data = self.df

        # Calculate mean and std along other dimensions if any
        # If it's already a 1D time series, mean/std won't do much
        dims_to_reduce = [d for d in data[self.y].dims if d != self.x]

        if dims_to_reduce:
            mean_data = data[self.y].mean(dim=dims_to_reduce)
            std_data = data[self.y].std(dim=dims_to_reduce)
        else:
            mean_data = data[self.y]
            std_data = xr.zeros_like(mean_data)

        plot_label = self.label if self.label is not None else self.y
        mean_data.plot(ax=self.ax, label=plot_label, **self.plotargs)

        upper = mean_data + std_data
        lower = mean_data - std_data

        self.ax.fill_between(
            mean_data[self.x].values, lower.values, upper.values, **self.fillargs
        )

        unit = data[self.y].attrs.get("units", "None")

        if self.ylabel is None:
            self.ax.set_ylabel(f"{self.y} ({unit})")
        else:
            self.ax.set_ylabel(self.ylabel)

        self.ax.set_xlabel(self.x)
        self.ax.legend()
        self.ax.set_title(self.title)
        self.fig.tight_layout()
        return self.ax


class TimeSeriesStatsPlot(BasePlot):
    """
    Create a time series plot of a specified statistic calculated between
    observations and model data, resampled to a given frequency.

    Supports lazy evaluation via xarray and dask.
    """

    def __init__(
        self,
        df: Any,
        col1: str,
        col2: Union[str, list[str]],
        x: Optional[str] = None,
        fig: Optional[matplotlib.figure.Figure] = None,
        ax: Optional[matplotlib.axes.Axes] = None,
        **kwargs: Any,
    ):
        """
        Initialize the TimeSeriesStatsPlot.

        Parameters
        ----------
        df : Any
            Data containing a time coordinate and the columns to compare.
            Can be pandas DataFrame, xarray Dataset, or xarray DataArray.
        col1 : str
            Name of the first column/variable (e.g., 'Obs').
        col2 : str or list of str
            Name of the second column(s)/variable(s) (e.g., 'Model').
        x : str, optional
            The time dimension/column name. If None, it attempts to find it
            automatically (prefers 'time' or 'datetime'), by default None.
        fig : matplotlib.figure.Figure, optional
            An existing Figure object.
        ax : matplotlib.axes.Axes, optional
            An existing Axes object.
        **kwargs : Any
            Additional arguments passed to BasePlot.
        """
        super().__init__(fig=fig, ax=ax, **kwargs)
        if self.ax is None:
            self.ax = self.fig.add_subplot(1, 1, 1)

        self.df = normalize_data(df)
        self.col1 = col1
        self.col2 = [col2] if isinstance(col2, str) else col2

        # Determine time coordinate/column
        if x is not None:
            self.x = x
        else:
            self.x = self._identify_time_coord()

        # Update history for provenance if xarray
        if isinstance(self.df, (xr.DataArray, xr.Dataset)):
            history = self.df.attrs.get("history", "")
            self.df.attrs["history"] = f"Initialized TimeSeriesStatsPlot; {history}"

    def _identify_time_coord(self) -> str:
        """
        Identify the time coordinate or column in the data.

        Returns
        -------
        str
            The identified time coordinate or column name.

        Raises
        ------
        ValueError
            If no suitable time coordinate or column is found.
        """
        if isinstance(self.df, (xr.DataArray, xr.Dataset)):
            for candidate in ["time", "datetime", "date"]:
                if candidate in self.df.coords or candidate in self.df.dims:
                    return candidate
            return self.df.dims[0]

        # Pandas
        if isinstance(self.df.index, pd.DatetimeIndex):
            return self.df.index.name if self.df.index.name else "index"
        for candidate in ["time", "datetime", "date"]:
            if candidate in self.df.columns:
                return candidate
        raise ValueError(
            "Could not identify time coordinate. Please specify 'x' parameter."
        )

    def plot(
        self, stat: str = "bias", freq: str = "D", **kwargs: Any
    ) -> matplotlib.axes.Axes:
        """
        Generate the time series plot for the chosen statistic.

        Parameters
        ----------
        stat : str, optional
            The statistic to calculate (e.g., 'bias', 'rmse', 'mae', 'corr').
            Supports any 'compute_<stat>' function in verification_metrics,
            by default "bias".
        freq : str, optional
            The resampling frequency (e.g., 'H', 'D', 'W', 'M'), by default "D".
        **kwargs : Any
            Keyword arguments passed to the plotting method.

        Returns
        -------
        matplotlib.axes.Axes
            The axes object with the plot.
        """
        from .. import verification_metrics

        stat_lower = stat.lower()
        metric_func = getattr(verification_metrics, f"compute_{stat_lower}", None)
        if metric_func is None:
            raise ValueError(f"Statistic '{stat}' is not supported.")

        plot_kwargs = {"marker": "o", "linestyle": "-"}
        plot_kwargs.update(kwargs)

        # Handle 'grid' separately as it's not a Line2D property
        show_grid = plot_kwargs.pop("grid", True)

        if isinstance(self.df, (xr.DataArray, xr.Dataset)):
            self._plot_xarray(metric_func, freq, stat_lower, plot_kwargs)
        else:
            self._plot_dataframe(metric_func, freq, stat_lower, plot_kwargs)

        if show_grid:
            self.ax.grid(True)

        self.ax.set_ylabel(stat.upper())
        self.ax.set_xlabel(self.x.capitalize())
        self.ax.legend()
        self.fig.tight_layout()

        # Update history for provenance
        if isinstance(self.df, (xr.DataArray, xr.Dataset)):
            history = self.df.attrs.get("history", "")
            self.df.attrs["history"] = (
                f"Generated TimeSeriesStatsPlot ({stat}, freq={freq}); {history}"
            )

        return self.ax

    def _plot_xarray(
        self, metric_func: Any, freq: str, stat_name: str, plot_kwargs: dict
    ) -> None:
        """
        Perform vectorized xarray/dask resampling and plotting.

        Parameters
        ----------
        metric_func : Any
            The metric function from verification_metrics to apply.
        freq : str
            The resampling frequency (e.g., 'D', 'H').
        stat_name : str
            The name of the statistic being calculated.
        plot_kwargs : dict
            Keyword arguments for the plot call.

        Examples
        --------
        >>> plot._plot_xarray(compute_bias, 'D', 'bias', {'color': 'red'})
        """
        for model_col in self.col2:

            def resample_func(ds):
                # Dim is None means reduce over all dimensions in the group
                # which is correct for a time series plot of a bulk statistic.
                return metric_func(ds[self.col1], ds[model_col])

            # Resample and calculate using .map() to maintain laziness
            resampled = self.df.resample({self.x: freq})
            stat_series = resampled.map(resample_func)

            # Extract label if present or use col name
            label = model_col
            stat_series.plot(ax=self.ax, label=label, **plot_kwargs)

    def _plot_dataframe(
        self, metric_func: Any, freq: str, stat_name: str, plot_kwargs: dict
    ) -> None:
        """
        Perform resampling and plotting for pandas DataFrames.

        Parameters
        ----------
        metric_func : Any
            The metric function from verification_metrics to apply.
        freq : str
            The resampling frequency (e.g., 'D', 'H').
        stat_name : str
            The name of the statistic being calculated.
        plot_kwargs : dict
            Keyword arguments for the plot call.

        Examples
        --------
        >>> plot._plot_dataframe(compute_bias, 'D', 'bias', {'marker': 'x'})
        """
        df = self.df.copy()
        if self.x != "index" and self.x in df.columns:
            df = df.set_index(self.x)

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        for model_col in self.col2:
            # Resample and apply metric
            # Note: Pandas resample.apply is less efficient but necessary here
            # for arbitrary metric functions on DataFrames.
            def pandas_metric(group):
                return metric_func(group[self.col1].values, group[model_col].values)

            stat_series = df.resample(freq).apply(pandas_metric)
            stat_series.plot(ax=self.ax, label=model_col, **plot_kwargs)
