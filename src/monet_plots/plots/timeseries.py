# src/monet_plots/plots/timeseries.py
import pandas as pd
from .base import BasePlot
from ..plot_utils import normalize_data
from typing import Any, Union, List, Optional
from .. import verification_metrics


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
        self.df = normalize_data(df)
        self.x = x
        self.y = y
        self.plotargs = plotargs
        self.fillargs = fillargs if fillargs is not None else {"alpha": 0.2}
        self.title = title
        self.ylabel = ylabel
        self.label = label

    def plot(self, **kwargs):
        """Generate the timeseries plot.

        Args:
            **kwargs: Overrides for plot settings (x, y, title, ylabel, label, etc.)
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

    def _plot_dataframe(self, **kwargs):
        """Generate the timeseries plot from pandas DataFrame."""
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

    def _plot_xarray(self, **kwargs):
        """Generate the timeseries plot from xarray DataArray or Dataset."""
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
    Create a time series plot of a specified statistic (e.g., bias, RMSE)
    calculated between two data columns, resampled to a given frequency.
    """

    def __init__(
        self, df: Any, col1: str, col2: Union[str, List[str]], *args, **kwargs
    ):
        """
        Initialize the plot with data.

        Args:
            df (pd.DataFrame, xr.Dataset, etc.): Data containing a time coordinate
                and the columns to compare.
            col1 (str): Name of the first column (e.g., 'Obs').
            col2 (str or list): Name of the second column(s)
                (e.g., 'Model' or ['Model1', 'Model2']).
            *args, **kwargs: Arguments passed to BasePlot.
        """
        super().__init__(*args, **kwargs)
        self.df = normalize_data(df)

        import xarray as xr

        if not isinstance(self.df, (xr.DataArray, xr.Dataset)):
            if not isinstance(self.df.index, pd.DatetimeIndex):
                # Attempt to set 'time' or 'datetime' column as index if not already
                if "datetime" in self.df.columns:
                    self.df = self.df.set_index("datetime")
                elif "time" in self.df.columns:
                    self.df = self.df.set_index("time")
                else:
                    # Try to convert index if it's not already datetime
                    try:
                        self.df.index = pd.to_datetime(self.df.index)
                    except Exception:
                        raise ValueError(
                            "Input DataFrame must have a DatetimeIndex "
                            "or 'time'/'datetime' column."
                        )

        self.col1 = col1
        if isinstance(col2, str):
            self.col2 = [col2]
        else:
            self.col2 = col2

    def plot(self, stat: str = "bias", freq: str = "D", **kwargs):
        """
        Generate the time series plot for the chosen statistic.

        Args:
            stat (str): The statistic to plot. Supported includes 'bias', 'rmse',
                'mae', 'corr', 'spearmanr', 'kendalltau', 'ioa', 'nse', 'kge',
                'mnb', 'mne', 'mape', 'mase', 'wdmb', 'stdo', 'stdp', 'r2'.
            freq (str): The resampling frequency (e.g., 'H', 'D', 'W', 'M').
            **kwargs: Keyword arguments passed to the plot() method.
        """
        stat_func_name = f"compute_{stat.lower()}"
        if not hasattr(verification_metrics, stat_func_name):
            msg = f"Statistic '{stat}' not supported."
            raise ValueError(msg)

        import xarray as xr

        if isinstance(self.df, (xr.DataArray, xr.Dataset)):
            return self._plot_xarray(stat.lower(), freq, **kwargs)
        else:
            return self._plot_dataframe(stat.lower(), freq, **kwargs)

    def _plot_dataframe(self, stat: str, freq: str, **kwargs):
        """Resample and plot using Pandas."""
        plot_kwargs = {"grid": True, "marker": "o", "linestyle": "-"}
        plot_kwargs.update(kwargs)

        func = getattr(verification_metrics, f"compute_{stat}")

        for model_col in self.col2:
            # Resampling stats on multiple columns is best done via groupby apply
            stat_series = self.df.groupby(pd.Grouper(freq=freq)).apply(
                lambda group: func(group[self.col1], group[model_col])
            )

            stat_series.plot(ax=self.ax, label=model_col, **plot_kwargs)

        self.ax.set_ylabel(f"{stat.upper()}")
        self.ax.set_xlabel("Date")
        self.ax.legend()
        self.fig.tight_layout()
        return self.ax

    def _plot_xarray(self, stat: str, freq: str, **kwargs):
        """Resample and plot using Xarray (Dask-friendly)."""
        import xarray as xr

        plot_kwargs = {"marker": "o", "linestyle": "-"}
        plot_kwargs.update(kwargs)

        # Detect time dimension
        time_dim = "time"
        if time_dim not in self.df.dims and "datetime" in self.df.dims:
            time_dim = "datetime"
        elif time_dim not in self.df.dims:
            # Try to find a dimension with time-like name or attributes
            for d in self.df.dims:
                if "time" in d.lower():
                    time_dim = d
                    break

        func = getattr(verification_metrics, f"compute_{stat}")

        ds = self.df
        for model_col in self.col2:
            # For all statistics in xarray resample, we use map with our vectorized metric
            # This preserves laziness if the inputs are dask-backed and ensures
            # we don't reimplement the logic here.
            stat_series = ds.resample({time_dim: freq}).map(
                lambda x: func(x[self.col1], x[model_col], dim=time_dim)
            )

            stat_series.plot(ax=self.ax, label=model_col, **plot_kwargs)

        self.ax.set_ylabel(f"{stat.upper()}")
        self.ax.set_xlabel("Date")
        self.ax.legend()
        self.fig.tight_layout()

        # Update history for provenance
        if isinstance(self.df, (xr.DataArray, xr.Dataset)):
            verification_metrics._update_history(
                self.df, f"Plotted {stat} time series resampled at {freq}"
            )

        return self.ax
