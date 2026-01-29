# src/monet_plots/plots/timeseries.py
import pandas as pd
import numpy as np
from .base import BasePlot
from ..plot_utils import normalize_data
from typing import Any, Union, List, Optional


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
                and the columns to compare. Must be convertible to a pandas
                DataFrame with a DatetimeIndex.
            col1 (str): Name of the first column (e.g., 'Obs').
            col2 (str or list): Name of the second column(s)
                (e.g., 'Model' or ['Model1', 'Model2']).
            *args, **kwargs: Arguments passed to BasePlot.
        """
        super().__init__(*args, **kwargs)
        if self.ax is None:
            self.ax = self.fig.add_subplot(1, 1, 1)
        self.df = normalize_data(df)
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
        self.stats = {
            "bias": self._calculate_bias,
            "rmse": self._calculate_rmse,
            "corr": self._calculate_corr,
        }

    def _calculate_bias(self, group, col2_name):
        """Calculate mean bias for a group."""
        return (group[col2_name] - group[self.col1]).mean()

    def _calculate_rmse(self, group, col2_name):
        """Calculate Root Mean Square Error for a group."""
        return np.sqrt(np.mean((group[col2_name] - group[self.col1]) ** 2))

    def _calculate_corr(self, group, col2_name):
        """Calculate Pearson correlation for a group."""
        return group[[self.col1, col2_name]].corr().iloc[0, 1]

    def plot(self, stat: str = "bias", freq: str = "D", **kwargs):
        """
        Generate the time series plot for the chosen statistic.

        Args:
            stat (str): The statistic to plot. Supported: 'bias', 'rmse', 'corr'.
            freq (str): The resampling frequency (e.g., 'H', 'D', 'W', 'M').
            **kwargs: Keyword arguments passed to the pandas plot() method.
        """
        if stat.lower() not in self.stats:
            msg = f"Statistic '{stat}' not supported. Use one of {list(self.stats.keys())}"
            raise ValueError(msg)

        plot_kwargs = {"grid": True, "marker": "o", "linestyle": "-"}
        plot_kwargs.update(kwargs)

        for model_col in self.col2:

            def stat_func(group):
                return self.stats[stat.lower()](group, model_col)

            stat_series = self.df.resample(freq).apply(stat_func)
            stat_series.plot(ax=self.ax, label=model_col, **plot_kwargs)

        self.ax.set_ylabel(f"{stat.upper()}")
        self.ax.set_xlabel("Date")
        self.ax.legend()
        self.fig.tight_layout()

        return self.ax
