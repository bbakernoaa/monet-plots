# src/monet_plots/plots/timeseries.py
import matplotlib.pyplot as plt
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

    def __init__(self, df: Any, x: str = "time", y: str = "obs", plotargs: dict = {}, fillargs: dict = {"alpha": 0.2}, title: str = "", ylabel: Optional[str] = None, label: Optional[str] = None, *args, **kwargs):
        """
        Initialize the plot with data and plot settings.

        Args:
            df (pd.DataFrame, np.ndarray, xr.Dataset, xr.DataArray): DataFrame with the data to plot.
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
        self.fillargs = fillargs
        self.title = title
        self.ylabel = ylabel
        self.label = label

    def plot(self, **kwargs):
        """Generate the timeseries plot."""
        import xarray as xr

        # Handle xarray objects differently from pandas DataFrames
        if isinstance(self.df, (xr.DataArray, xr.Dataset)):
            return self._plot_xarray(**kwargs)
        else:
            return self._plot_dataframe(**kwargs)

    def _plot_dataframe(self, **kwargs):
        """Generate the timeseries plot from pandas DataFrame (original implementation)."""
        self.df.index = self.df[self.x]
        df = self.df.drop(columns=self.x).reset_index()
        m = df.groupby("time").mean(numeric_only=True)
        e = df.groupby("time").std(numeric_only=True)
        variable = self.y
        if df.columns.isin(["units"]).max():
            unit = df.units
        else:
            unit = "None"
        upper = m[self.y] + e[self.y]
        lower = m[self.y] - e[self.y]
        lower.loc[lower < 0] = 0
        lower = lower.values
        if "alpha" not in self.fillargs:
            self.fillargs["alpha"] = 0.2
        if self.label is not None:
            m.rename(columns={self.y: self.label}, inplace=True)
        else:
            self.label = self.y
        m[self.label].plot(ax=self.ax, **self.plotargs)
        self.ax.fill_between(m[self.label].index, lower, upper, **self.fillargs)
        if self.ylabel is None:
            self.ax.set_ylabel(variable + " (" + unit + ")")
        else:
            self.ax.set_ylabel(self.label)
        self.ax.set_xlabel("")
        self.ax.legend()
        self.ax.set_title(self.title)
        self.fig.tight_layout()
        return self.ax

    def _plot_xarray(self, **kwargs):
        """Generate the timeseries plot from xarray DataArray or Dataset."""
        import xarray as xr

        # Ensure we have the right data structure
        if isinstance(self.df, xr.DataArray):
            # Convert DataArray to Dataset for easier handling
            data = self.df.to_dataset(name=self.y)
        else:
            data = self.df

        # Set default fill alpha if not provided
        if "alpha" not in self.fillargs:
            self.fillargs["alpha"] = 0.2

        # Use xarray's built-in plotting capabilities
        # For time series, we can use the plot method with error shading
        if self.label is not None:
            data[self.y].plot(ax=self.ax, label=self.label, **self.plotargs)
        else:
            data[self.y].plot(ax=self.ax, **self.plotargs)

        # Calculate and plot error bounds
        # Use the time coordinate directly from the data
        time_coord = data[self.y].coords[self.x]
        mean_data = data[self.y].mean(dim=self.x)
        std_data = data[self.y].std(dim=self.x)
        upper = mean_data + std_data
        lower = mean_data - std_data
        lower = lower.where(lower >= 0, 0)  # Ensure non-negative values

        # Plot the error bounds using the time coordinate values
        self.ax.fill_between(
            time_coord.values,
            lower.values,
            upper.values,
            **self.fillargs
        )

        # Set labels and title
        unit = "None"  # xarray doesn't have a direct units attribute like pandas
        if hasattr(data[self.y], 'attrs') and 'units' in data[self.y].attrs:
            unit = data[self.y].attrs['units']

        if self.ylabel is None:
            self.ax.set_ylabel(f"{self.y} ({unit})")
        else:
            self.ax.set_ylabel(self.label)

        self.ax.set_xlabel("")
        self.ax.legend()
        self.ax.set_title(self.title)
        self.fig.tight_layout()
        return self.ax


class TimeSeriesStatsPlot(BasePlot):
    """
    Create a time series plot of a specified statistic (e.g., bias, RMSE)
    calculated between two data columns, resampled to a given frequency.
    """

    def __init__(self, df: Any, col1: str, col2: Union[str, List[str]], *args, **kwargs):
        """
        Initialize the plot with data.

        Args:
            df (pd.DataFrame, xr.Dataset, etc.): Data containing a time coordinate
                and the columns to compare. Must be convertible to a pandas
                DataFrame with a DatetimeIndex.
            col1 (str): Name of the first column (e.g., 'Obs').
            col2 (str or list): Name of the second column(s) (e.g., 'Model' or ['Model1', 'Model2']).
            *args, **kwargs: Arguments passed to BasePlot.
        """
        super().__init__(*args, **kwargs)
        self.df = normalize_data(df)
        if not isinstance(self.df.index, pd.DatetimeIndex):
            # Attempt to set 'time' or 'datetime' column as index if not already
            if "datetime" in self.df.columns:
                self.df = self.df.set_index("datetime")
            elif "time" in self.df.columns:
                self.df = self.df.set_index("time")
            else:
                raise ValueError("Input DataFrame must have a DatetimeIndex.")

        self.col1 = col1
        if isinstance(col2, str):
            self.col2 = [col2]  # Internally, always treat col2 as a list
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

    def _calculate_corr(self, group):
        """Calculate Pearson correlation for a group."""
        return group[[self.col1, self.col2]].corr().iloc[0, 1]

    def plot(self, stat: str = "bias", freq: str = "D", **kwargs):
        """
        Generate the time series plot for the chosen statistic.

        Args:
            stat (str): The statistic to plot. Supported: 'bias', 'rmse', 'corr'.
            freq (str): The resampling frequency (e.g., 'H', 'D', 'W', 'M').
                        See pandas frequency strings for options.
            **kwargs: Keyword arguments passed to the pandas plot() method.

        Returns:
            matplotlib.axes.Axes: The axes object containing the plot.
        """
        if stat.lower() not in self.stats:
            raise ValueError(f"Statistic '{stat}' not supported. Use one of {list(self.stats.keys())}")

        # Set default plot properties, allowing user to override
        plot_kwargs = {
            "grid": True,
            "marker": "o",
            "linestyle": "-"
        }
        # User-provided kwargs will override defaults but not the label
        plot_kwargs = {**plot_kwargs, **kwargs}

        for model_col in self.col2:
            # Define a lambda to pass the current model column to the stat function
            stat_func = lambda group: self.stats[stat.lower()](group, model_col)

            # Resample and apply the chosen statistical function for the current model
            stat_series = self.df.resample(freq).apply(stat_func)

            # Plot the series with the model column name as the label
            stat_series.plot(ax=self.ax, label=model_col, **plot_kwargs)

        self.ax.set_ylabel(f"{stat.upper()}")
        self.ax.set_xlabel("Date")
        self.ax.legend()
        self.fig.tight_layout()

        return self.ax
