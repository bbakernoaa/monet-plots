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
        data: Any = None,
        x: str = "time",
        y: str = "obs",
        plotargs: dict = {},
        fillargs: dict = None,
        title: str = "",
        ylabel: Optional[str] = None,
        label: Optional[str] = None,
        *args,
        df: Any = None,
        **kwargs,
    ):
        """
        Initialize the plot with data and plot settings.

        Args:
            data (pd.DataFrame, np.ndarray, xr.Dataset, xr.DataArray):
                Data to plot.
            x (str): Column name for the x-axis (time).
            y (str): Column name for the y-axis (values).
            plotargs (dict): Arguments for the plot.
            fillargs (dict): Arguments for fill_between.
            title (str): Title for the plot.
            ylabel (str, optional): Y-axis label.
            label (str, optional): Label for the plotted line.
            df: Deprecated alias for ``data``.
            *args, **kwargs: Arguments passed to BasePlot.
        """
        super().__init__(*args, **kwargs)
        if self.ax is None:
            self.ax = self.fig.add_subplot(1, 1, 1)

        if df is not None and data is None:
            data = df
        self.data = normalize_data(data, prefer_xarray=False)
        self.x = x
        self.y = y
        self.plotargs = plotargs
        self.fillargs = fillargs if fillargs is not None else {"alpha": 0.2}
        self.title = title
        self.ylabel = ylabel
        self.label = label

    def plot(self, **kwargs: Any) -> plt.Axes:
        """
        Generate the timeseries plot.

        Parameters
        ----------
        **kwargs : Any
            Overrides for plot settings (x, y, title, ylabel, label, etc.).

        Returns
        -------
        plt.Axes
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
        if isinstance(self.data, (xr.DataArray, xr.Dataset)):
            return self._plot_xarray(**kwargs)
        else:
            return self._plot_dataframe(**kwargs)

    def _plot_dataframe(self, **kwargs: Any) -> plt.Axes:
        """
        Generate the timeseries plot from pandas DataFrame.

        Parameters
        ----------
        **kwargs : Any
            Additional plotting arguments.

        Returns
        -------
        plt.Axes
            The matplotlib axes object.

        Examples
        --------
        >>> plot._plot_dataframe()
        """
        df = self.data.copy()
        df.index = df[self.x]
        # Keep only numeric columns for grouping, but make sure self.y is there
        df = df.reset_index(drop=True)
        # We need to preserve self.x for grouping if it's not the index
        m = self.data.groupby(self.x).mean(numeric_only=True)
        e = self.data.groupby(self.x).std(numeric_only=True)

        variable = self.y
        unit = "None"
        if "units" in self.data.columns:
            unit = str(self.data["units"].iloc[0])

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
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.fig.tight_layout()
        return self.ax

    def _plot_xarray(self, **kwargs: Any) -> plt.Axes:
        """
        Generate the timeseries plot from xarray DataArray or Dataset.

        Parameters
        ----------
        **kwargs : Any
            Additional plotting arguments.

        Returns
        -------
        plt.Axes
            The matplotlib axes object.

        Examples
        --------
        >>> plot._plot_xarray()
        """
        import xarray as xr

        # Ensure we have the right data structure
        if isinstance(self.data, xr.DataArray):
            data = (
                self.data.to_dataset(name=self.y)
                if self.data.name is None
                else self.data.to_dataset()
            )
            if self.data.name is not None:
                self.y = self.data.name
        else:
            data = self.data

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
        mean_data.dropna(self.x).plot(ax=self.ax, label=plot_label, **self.plotargs)

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
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.fig.tight_layout()
        return self.ax


class TimeSeriesStatsPlot(BasePlot):
    """
    Create a time series plot of a specified statistic calculated between
    var1 and var2, resampled to a given frequency (Unified API).

    Supports lazy evaluation via xarray and dask.

    Parameters
    ----------
    data : Any
        Data containing a time coordinate and the columns to compare.
    var1 : str
        Name of the first variable (e.g., observations).
    var2 : str or list of str
        Name(s) of the second variable(s) (e.g., model(s)).
    x : str, optional
        The time dimension/column name. If None, it attempts to find it
        automatically (prefers 'time' or 'datetime'), by default None.
    fig : matplotlib.figure.Figure, optional
        An existing Figure object.
    ax : matplotlib.axes.Axes, optional
        An existing Axes object.
    df : Any, optional
        Deprecated alias for ``data``.
    col1 : str, optional
        Legacy alias for var1.
    col2 : str or list, optional
        Legacy alias for var2.
    **kwargs : Any
        Additional arguments passed to BasePlot.
    """

    def __init__(
        self,
        data: Any = None,
        var1: str = None,
        var2: Union[str, list[str]] = None,
        x: Optional[str] = None,
        fig: Optional[matplotlib.figure.Figure] = None,
        ax: Optional[matplotlib.axes.Axes] = None,
        df: Any = None,
        col1: str = None,  # legacy alias
        col2: Union[str, list[str]] = None,  # legacy alias
        **kwargs: Any,
    ):
        super().__init__(fig=fig, ax=ax, **kwargs)
        if self.ax is None:
            self.ax = self.fig.add_subplot(1, 1, 1)

        if df is not None and data is None:
            data = df
        self.data = normalize_data(data)
        self.var1 = var1 or col1
        self.col1 = self.var1
        if var2 is not None:
            self.var2 = [var2] if isinstance(var2, str) else var2
        elif col2 is not None:
            self.var2 = [col2] if isinstance(col2, str) else col2
        else:
            self.var2 = None
        self.col2 = self.var2

        # Determine time coordinate/column
        if x is not None:
            self.x = x
        else:
            self.x = self._identify_time_coord()

        # Update history for provenance if xarray
        if isinstance(self.data, (xr.DataArray, xr.Dataset)):
            history = self.data.attrs.get("history", "")
            self.data.attrs["history"] = f"Initialized TimeSeriesStatsPlot; {history}"

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
        if isinstance(self.data, (xr.DataArray, xr.Dataset)):
            for candidate in ["time", "datetime", "date"]:
                if candidate in self.data.coords or candidate in self.data.dims:
                    return candidate
            if self.data.dims:
                return str(self.data.dims[0])
            raise ValueError("Could not identify time dimension in xarray object.")

        # Pandas
        if isinstance(self.data.index, pd.DatetimeIndex):
            return self.data.index.name if self.data.index.name else "index"
        for candidate in ["time", "datetime", "date"]:
            if candidate in self.data.columns:
                return candidate
        raise ValueError(
            "Could not identify time coordinate. Please specify 'x' parameter."
        )

    def plot(self, stat: str = "bias", freq: str = "D", **kwargs: Any) -> plt.Axes:
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

        if isinstance(self.data, (xr.DataArray, xr.Dataset)):
            self._plot_xarray(metric_func, freq, stat_lower, plot_kwargs)
        else:
            self._plot_dataframe(metric_func, freq, stat_lower, plot_kwargs)

        if show_grid:
            self.ax.grid(True)

        self.ax.set_ylabel(stat.upper())
        self.ax.set_xlabel(self.x.capitalize())
        self.ax.legend()
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.fig.tight_layout()

        # Update history for provenance
        if isinstance(self.data, (xr.DataArray, xr.Dataset)):
            history = self.data.attrs.get("history", "")
            self.data.attrs["history"] = (
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
        for model_col in self.var2:

            def resample_func(ds):
                # Dim is None means reduce over all dimensions in the group
                # which is correct for a time series plot of a bulk statistic.
                return metric_func(ds[self.col1], ds[model_col])

            # Resample and calculate using .map() to maintain laziness
            resampled = self.data.resample({self.x: freq})
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
        df = self.data.copy()
        if self.x != "index" and self.x in df.columns:
            df = df.set_index(self.x)

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        for model_col in self.var2:
            # Resample and apply metric
            # Note: Pandas resample.apply is less efficient but necessary here
            # for arbitrary metric functions on DataFrames.
            def pandas_metric(group):
                return metric_func(group[self.col1].values, group[model_col].values)

            stat_series = df.resample(freq).apply(pandas_metric)
            stat_series.plot(ax=self.ax, label=model_col, **plot_kwargs)


class TimeSeriesErrorBarPlot(BasePlot):
    """Time series plot with discrete error bars.

    Plots the mean of one or more variables over time with ±1 standard
    deviation (or explicit error values) shown as error bars.  Unlike
    :class:`TimeSeriesPlot`, which uses a continuous shaded fill, this class
    uses ``ax.errorbar`` so individual bars are visible at each time step.

    Typical use cases include comparing model vs. observation means at each
    forecast cycle, or visualising ensemble spread at discrete lead times.
    """

    def __init__(
        self,
        data: Any = None,
        x: str = "time",
        y: Union[str, list[str]] = "obs",
        *,
        yerr: Optional[Union[str, list[str]]] = None,
        freq: Optional[str] = None,
        label_col: Optional[str] = None,
        title: str = "",
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        df: Any = None,
        **kwargs: Any,
    ) -> None:
        """Initialise the error-bar time series plot.

        Parameters
        ----------
        data : Any
            Input data. Accepts a pandas DataFrame, xarray Dataset/DataArray,
            or numpy array.
        x : str, optional
            Column / coordinate name for the time axis, by default ``"time"``.
        y : str or list of str, optional
            Column(s) to plot on the y-axis, by default ``"obs"``.  Each
            column produces one error-bar series.
        yerr : str or list of str, optional
            Column(s) containing the pre-computed error for each ``y`` column.
            When *None* (default) and ``freq`` is set, the standard deviation
            within each resampled bin is used.  When *None* and ``freq`` is
            *None*, the standard deviation across repeated time values is used.
        freq : str, optional
            Pandas/xarray resampling frequency (e.g. ``"D"``, ``"6h"``).  When
            provided the data are resampled to this frequency before computing
            mean and std, by default *None* (no resampling).
        label_col : str, optional
            Column whose unique values are used to split the data into separate
            series (e.g. a ``"model"`` column), by default *None*.
        title : str, optional
            Plot title, by default ``""``.
        xlabel : str, optional
            Override for the x-axis label, by default *None* (uses ``x``).
        ylabel : str, optional
            Override for the y-axis label, by default *None* (uses ``y``).
        df : Any, optional
            Deprecated alias for ``data``.
        **kwargs : Any
            Forwarded to :class:`BasePlot`.
        """
        super().__init__(**kwargs)
        if self.ax is None:
            self.ax = self.fig.add_subplot(1, 1, 1)

        if df is not None and data is None:
            data = df
        self.data = normalize_data(data, prefer_xarray=False)
        self.x = x
        self.y = [y] if isinstance(y, str) else list(y)
        self.yerr = (
            ([yerr] if isinstance(yerr, str) else list(yerr))
            if yerr is not None
            else None
        )
        self.freq = freq
        self.label_col = label_col
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def plot(
        self,
        capsize: int = 4,
        fmt: str = "o-",
        **kwargs: Any,
    ) -> plt.Axes:
        """Generate the error-bar time series plot.

        Parameters
        ----------
        capsize : int, optional
            Length of the error-bar caps in points, by default ``4``.
        fmt : str, optional
            Format string passed to ``ax.errorbar``, by default ``"o-"``.
        **kwargs : Any
            Additional keyword arguments forwarded to ``ax.errorbar``.

        Returns
        -------
        matplotlib.axes.Axes
            The axes object containing the plot.
        """
        if isinstance(self.data, (xr.DataArray, xr.Dataset)):
            self._plot_xarray(capsize=capsize, fmt=fmt, **kwargs)
        else:
            self._plot_dataframe(capsize=capsize, fmt=fmt, **kwargs)

        self.ax.set_xlabel(self.xlabel or self.x)
        self.ax.set_ylabel(self.ylabel or (self.y[0] if len(self.y) == 1 else "value"))
        self.ax.set_title(self.title)
        self.ax.legend()

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.fig.tight_layout()

        return self.ax

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _plot_dataframe(self, capsize: int, fmt: str, **kwargs: Any) -> None:
        """Plot from a pandas DataFrame."""
        df = self.data.copy()

        # Ensure time column is available as a column (not only as index)
        if self.x == "index" or self.x not in df.columns:
            df[self.x] = df.index

        if not pd.api.types.is_datetime64_any_dtype(df[self.x]):
            df[self.x] = pd.to_datetime(df[self.x])

        groups = (
            [("_all_", df)]
            if self.label_col is None
            else [(lbl, grp) for lbl, grp in df.groupby(self.label_col)]
        )

        for grp_label, grp_df in groups:
            for idx, col in enumerate(self.y):
                label = (
                    col
                    if grp_label == "_all_"
                    else f"{grp_label} – {col}"
                    if len(self.y) > 1
                    else str(grp_label)
                )

                if self.freq is not None:
                    tmp = grp_df.set_index(self.x)[col]
                    if not isinstance(tmp.index, pd.DatetimeIndex):
                        tmp.index = pd.to_datetime(tmp.index)
                    mean_s = tmp.resample(self.freq).mean()
                    err_s = tmp.resample(self.freq).std().fillna(0)
                    times, means, errs = mean_s.index, mean_s.values, err_s.values
                elif self.yerr is not None:
                    err_col = self.yerr[idx] if idx < len(self.yerr) else self.yerr[-1]
                    agg = grp_df.groupby(self.x).agg(
                        _mean=(col, "mean"), _err=(err_col, "mean")
                    )
                    times, means, errs = (
                        agg.index,
                        agg["_mean"].values,
                        agg["_err"].values,
                    )
                else:
                    agg = grp_df.groupby(self.x)[col].agg(["mean", "std"]).fillna(0)
                    times, means, errs = (
                        agg.index,
                        agg["mean"].values,
                        agg["std"].values,
                    )

                self.ax.errorbar(
                    times,
                    means,
                    yerr=errs,
                    label=label,
                    fmt=fmt,
                    capsize=capsize,
                    **kwargs,
                )

    def _plot_xarray(self, capsize: int, fmt: str, **kwargs: Any) -> None:
        """Plot from an xarray Dataset or DataArray."""
        if isinstance(self.data, xr.DataArray):
            ds = self.data.to_dataset(name=self.data.name or self.y[0])
        else:
            ds = self.data

        for col in self.y:
            da = ds[col]
            time_dim = self.x if self.x in da.dims else da.dims[0]

            if self.freq is not None:
                mean_da = da.resample({time_dim: self.freq}).mean()
                err_da = da.resample({time_dim: self.freq}).std().fillna(0)
            else:
                # Reduce any non-time dimensions
                other_dims = [d for d in da.dims if d != time_dim]
                mean_da = da.mean(dim=other_dims) if other_dims else da
                err_da = (
                    da.std(dim=other_dims).fillna(0)
                    if other_dims
                    else xr.zeros_like(da)
                )

            times = mean_da[time_dim].values
            means = mean_da.values
            errs = err_da.values

            self.ax.errorbar(
                times, means, yerr=errs, label=col, fmt=fmt, capsize=capsize, **kwargs
            )
