# src/monet_plots/plots/timeseries.py
import pandas as pd
import numpy as np
from .base import BasePlot
from ..plot_utils import normalize_data
from typing import Any, Union, List, Optional


class TimeSeriesPlot(BasePlot):
    """Create a timeseries plot with shaded error bounds.

    This plot supports both pandas DataFrames and xarray Datasets.
    """

    def __init__(
        self,
        data: Any,
        x: str = "time",
        y: str = "obs",
        freq: str = "D",
        plotargs: dict = {},
        fillargs: dict = {"alpha": 0.2},
        title: str = "",
        ylabel: Optional[str] = None,
        label: Optional[str] = None,
        *args,
        **kwargs,
    ):
        """
        Initialize the plot with data and settings.

        Args:
            data (pd.DataFrame, xr.Dataset): Data to plot.
            x (str): Name of the time coordinate.
            y (str): Name of the data variable.
            freq (str): Resampling frequency.
            plotargs (dict): Keyword arguments for the line plot.
            fillargs (dict): Keyword arguments for the shaded error bounds.
            title (str): Plot title.
            ylabel (str, optional): Y-axis label.
            label (str, optional): Legend label for the line.
        """
        super().__init__(*args, **kwargs)
        self.data = normalize_data(data)
        self.x = x
        self.y = y
        self.freq = freq
        self.plotargs = plotargs
        self.fillargs = fillargs
        self.title = title
        self.ylabel = ylabel
        self.label = label if label is not None else y

    def plot(self, **kwargs):
        """Generate the timeseries plot."""
        if isinstance(self.data, pd.DataFrame):
            self._plot_dataframe(**kwargs)
        else:
            self._plot_xarray(**kwargs)

        self.ax.set_title(self.title)
        self.ax.set_xlabel("")
        self.ax.legend()
        self.fig.tight_layout()
        return self.ax

    def _plot_dataframe(self, **kwargs):
        """Plotting logic for pandas DataFrame."""
        df_resampled = self.data.set_index(self.x)
        mean = df_resampled[self.y].resample(self.freq).mean()
        std = df_resampled[self.y].resample(self.freq).std()

        mean.plot(ax=self.ax, label=self.label, **self.plotargs)
        self.ax.fill_between(mean.index, mean - std, mean + std, **self.fillargs)

        unit = self.data.get("units", "None")
        if self.ylabel is None:
            self.ax.set_ylabel(f"{self.y} ({unit})")
        else:
            self.ax.set_ylabel(self.ylabel)

    def _plot_xarray(self, **kwargs):
        """Plotting logic for xarray Dataset."""
        mean = self.data[self.y].resample({self.x: self.freq}).mean()
        std = self.data[self.y].resample({self.x: self.freq}).std()

        mean.plot(ax=self.ax, label=self.label, **self.plotargs)
        self.ax.fill_between(mean[self.x].values, (mean - std).values, (mean + std).values, **self.fillargs)

        unit = self.data[self.y].attrs.get("units", "None")
        if self.ylabel is None:
            self.ax.set_ylabel(f"{self.y} ({unit})")
        else:
            self.ax.set_ylabel(self.ylabel)


class TimeSeriesStatsPlot(BasePlot):
    """
    Create a time series plot of a specified statistic (e.g., bias, RMSE)
    calculated between two data columns, resampled to a given frequency.
    """

    def __init__(self, data: Any, col1: str, col2: Union[str, List[str]], *args, **kwargs):
        """
        Initialize the plot with data.

        Args:
            data (pd.DataFrame, xr.Dataset): Data containing a time coordinate
                and the columns to compare.
            col1 (str): Name of the first column/variable (e.g., 'Obs').
            col2 (str or list): Name of the second column(s)/variable(s) (e.g., 'Model').
        """
        super().__init__(*args, **kwargs)
        self.data = normalize_data(data)
        self.col1 = col1
        self.col2 = [col2] if isinstance(col2, str) else col2

    def _calculate_stats_pd(self, df, stat, freq):
        """Calculate statistics for pandas DataFrame."""
        results = {}
        for model_col in self.col2:
            if stat == 'bias':
                stat_series = (df[model_col] - df[self.col1]).resample(freq).mean()
            elif stat == 'rmse':
                stat_series = ((df[model_col] - df[self.col1])**2).resample(freq).mean().apply(np.sqrt)
            elif stat == 'corr':
                stat_series = df[[self.col1, model_col]].resample(freq).corr().unstack()[self.col1][model_col]
            results[model_col] = stat_series
        return results

    def _calculate_stats_xr(self, ds, stat, freq):
        """Calculate statistics for xarray Dataset."""
        results = {}
        for model_var in self.col2:
            if stat == 'bias':
                stat_da = (ds[model_var] - ds[self.col1]).resample(time=freq).mean()
            elif stat == 'rmse':
                stat_da = ((ds[model_var] - ds[self.col1])**2).resample(time=freq).mean()**0.5
            elif stat == 'corr':
                # Xarray doesn't have a direct resampling correlation, so we use a workaround.
                # This is a bit more complex and might need a dedicated library for efficiency.
                # For now, we convert to DataFrame for this specific calculation.
                df = ds[[self.col1, model_var]].to_dataframe()
                stat_series = df.resample(freq).corr().unstack()[self.col1][model_var]
                results[model_var] = stat_series
                continue
            results[model_var] = stat_da
        return results

    def plot(self, stat: str = "bias", freq: str = "D", **kwargs):
        """
        Generate the time series plot for the chosen statistic.

        Args:
            stat (str): The statistic to plot. Supported: 'bias', 'rmse', 'corr'.
            freq (str): The resampling frequency (e.g., 'H', 'D', 'W', 'M').
            **kwargs: Keyword arguments passed to the plot() method.
        """
        if stat.lower() not in ['bias', 'rmse', 'corr']:
            raise ValueError(f"Statistic '{stat}' not supported. Use 'bias', 'rmse', or 'corr'.")

        plot_kwargs = {"marker": "o", "linestyle": "-"}
        plot_kwargs.update(kwargs)
        grid = plot_kwargs.pop("grid", True)

        if isinstance(self.data, pd.DataFrame):
            # Ensure 'time' is the index
            if 'time' in self.data.columns:
                self.data = self.data.set_index('time')
            if not isinstance(self.data.index, pd.DatetimeIndex):
                raise ValueError("DataFrame must have a DatetimeIndex or a 'time' column.")

            stats_results = self._calculate_stats_pd(self.data, stat, freq)
            for model_col, stat_series in stats_results.items():
                stat_series.plot(ax=self.ax, label=model_col, **plot_kwargs)
        else: # xarray.Dataset
            stats_results = self._calculate_stats_xr(self.data, stat, freq)
            for model_var, stat_da in stats_results.items():
                stat_da.plot(ax=self.ax, label=model_var, **plot_kwargs)

        if grid:
            self.ax.grid()

        self.ax.set_ylabel(f"{stat.upper()}")
        self.ax.set_xlabel("Date")
        self.ax.legend()
        self.fig.tight_layout()
        return self.ax
