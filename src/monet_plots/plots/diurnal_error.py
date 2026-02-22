# src/monet_plots/plots/diurnal_error.py
"""Diurnal error heat map for model performance analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr

from ..plot_utils import _update_history, normalize_data
from .base import BasePlot

if TYPE_CHECKING:
    import holoviews as hv
    import matplotlib.axes
    import matplotlib.figure


class DiurnalErrorPlot(BasePlot):
    """Diurnal error heat map.

    Visualizes model error (bias) as a function of the hour of day and another
    temporal dimension (e.g., month, day of week, or date).

    This class supports native Xarray and Dask objects for lazy evaluation
    and provenance tracking.

    Attributes
    ----------
    data : Union[xr.Dataset, xr.DataArray, pd.DataFrame]
        The input data for the plot.
    obs_col : str
        Column/variable name for observations.
    mod_col : str
        Column/variable name for model values.
    time_col : str
        Dimension/column name for timestamp.
    second_dim : str
        The second dimension for the heatmap.
    metric : str
        The metric to plot ('bias' or 'error').
    aggregated : xr.DataArray
        The calculated aggregated data for the heatmap.
    second_label : str
        The label for the second dimension on the y-axis.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from monet_plots.plots import DiurnalErrorPlot
    >>> dates = pd.date_range("2023-01-01", periods=100, freq="h")
    >>> df = pd.DataFrame({
    ...     "time": dates,
    ...     "obs": np.random.rand(100),
    ...     "mod": np.random.rand(100)
    ... })
    >>> plot = DiurnalErrorPlot(df, obs_col="obs", mod_col="mod")
    >>> ax = plot.plot()
    """

    def __init__(
        self,
        data: Any,
        obs_col: str,
        mod_col: str,
        *,
        time_col: str = "time",
        second_dim: str = "month",
        metric: str = "bias",
        fig: matplotlib.figure.Figure | None = None,
        ax: matplotlib.axes.Axes | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize Diurnal Error Plot.

        Parameters
        ----------
        data : Any
            Input data. Can be a pandas DataFrame, xarray DataArray,
            xarray Dataset, or dask-backed object.
        obs_col : str
            Column/variable name for observations.
        mod_col : str
            Column/variable name for model values.
        time_col : str, optional
            Dimension/column name for timestamp, by default "time".
        second_dim : str, optional
            The second dimension for the heatmap ('month', 'dayofweek', 'date',
            or a coordinate name), by default "month".
        metric : str, optional
            The metric to plot ('bias' or 'error'), by default "bias".
        fig : matplotlib.figure.Figure, optional
            Existing figure object, by default None.
        ax : matplotlib.axes.Axes, optional
            Existing axes object, by default None.
        **kwargs : Any
            Additional arguments passed to BasePlot.
        """
        super().__init__(fig=fig, ax=ax, **kwargs)

        # Normalize data to Xarray if possible
        self.data = normalize_data(data)
        self.obs_col = obs_col
        self.mod_col = mod_col
        self.time_col = time_col
        self.second_dim = second_dim
        self.metric = metric
        self.aggregated: xr.DataArray | None = None
        self.second_label: str = ""

        # Prepare the calculation
        self._calculate_metric()

    def _calculate_metric(self) -> None:
        """Calculates the aggregated metric for the heatmap.

        This method identifies the appropriate backend (Xarray/Dask or Pandas),
        calculates the specified metric (bias or absolute error), and aggregates
        it into a 2D grid indexed by 'second_val' and 'hour'. It maintains
        lazy evaluation for Dask-backed objects and uses vectorized grouping.

        Raises
        ------
        ValueError
            If the metric is not 'bias' or 'error', or if second_dim is not found.
        """
        # Convert to Dataset if it's a DataArray to handle multiple columns easily
        ds = self.data
        if isinstance(ds, xr.DataArray):
            ds = ds.to_dataset()

        if isinstance(ds, xr.Dataset):
            # Calculate individual error/bias lazily
            if self.metric == "bias":
                val = ds[self.mod_col] - ds[self.obs_col]
                val.name = "bias"
                msg = "Calculated diurnal bias"
            elif self.metric == "error":
                val = np.abs(ds[self.mod_col] - ds[self.obs_col])
                val.name = "error"
                msg = "Calculated diurnal absolute error"
            else:
                raise ValueError("metric must be 'bias' or 'error'")

            # Add temporal coordinates for grouping
            time_coord = ds[self.time_col]
            val = val.assign_coords(hour=time_coord.dt.hour)

            if self.second_dim == "month":
                val = val.assign_coords(second_val=time_coord.dt.month)
                self.second_label = "Month"
            elif self.second_dim == "dayofweek":
                val = val.assign_coords(second_val=time_coord.dt.dayofweek)
                self.second_label = "Day of Week"
            elif self.second_dim == "date":
                val = val.assign_coords(second_val=time_coord.dt.floor("D"))
                self.second_label = "Date"
            else:
                if self.second_dim in ds.coords or self.second_dim in ds.data_vars:
                    val = val.assign_coords(second_val=ds[self.second_dim])
                    self.second_label = self.second_dim
                else:
                    raise ValueError(
                        f"second_dim '{self.second_dim}' not found in data"
                    )

            # Aero Protocol: Vectorized multi-dimensional grouping.
            # This is significantly faster than the previous loop and maintains
            # lazy evaluation for Dask-backed objects.
            try:
                self.aggregated = val.groupby(["second_val", "hour"]).mean(
                    dim=self.time_col, keep_attrs=True
                )

                # Ensure consistent order (second_val, hour)
                if (
                    "second_val" in self.aggregated.dims
                    and "hour" in self.aggregated.dims
                ):
                    self.aggregated = self.aggregated.transpose("second_val", "hour")
            except Exception:
                # Fallback to eager if something goes wrong with complex Xarray ops
                # though modern xarray (with flox) should handle this lazily.
                df = val.to_dataframe(name=val.name).reset_index()
                pivot = df.pivot_table(
                    index="second_val", columns="hour", values=val.name, aggfunc="mean"
                )
                self.aggregated = xr.DataArray(
                    pivot.values,
                    coords={
                        "second_val": pivot.index.values,
                        "hour": pivot.columns.values,
                    },
                    dims=["second_val", "hour"],
                    name=val.name,
                )

            self.aggregated = _update_history(self.aggregated, msg)

        else:
            # Fallback for Pandas DataFrame (backward compatibility)
            df = self.data.copy()
            df[self.time_col] = pd.to_datetime(df[self.time_col])
            df["hour"] = df[self.time_col].dt.hour

            if self.second_dim == "month":
                df["second_val"] = df[self.time_col].dt.month
                self.second_label = "Month"
            elif self.second_dim == "dayofweek":
                df["second_val"] = df[self.time_col].dt.dayofweek
                self.second_label = "Day of Week"
            elif self.second_dim == "date":
                df["second_val"] = df[self.time_col].dt.date
                self.second_label = "Date"
            else:
                df["second_val"] = df[self.second_dim]
                self.second_label = self.second_dim

            if self.metric == "bias":
                df["val"] = df[self.mod_col] - df[self.obs_col]
                metric_name = "bias"
                msg = "Calculated diurnal bias"
            elif self.metric == "error":
                df["val"] = np.abs(df[self.mod_col] - df[self.obs_col])
                metric_name = "error"
                msg = "Calculated diurnal absolute error"
            else:
                df["val"] = df[self.mod_col]
                metric_name = "value"
                msg = "Calculated diurnal values"

            pivot = df.pivot_table(
                index="second_val", columns="hour", values="val", aggfunc="mean"
            )
            self.aggregated = xr.DataArray(
                pivot.values,
                coords={
                    "second_val": pivot.index.values,
                    "hour": pivot.columns.values,
                },
                dims=["second_val", "hour"],
                name=metric_name,
            )
            self.aggregated = _update_history(self.aggregated, msg)

    def plot(self, cmap: str = "RdBu_r", **kwargs: Any) -> matplotlib.axes.Axes:
        """
        Generate the diurnal error heatmap (Track A: Static).

        Parameters
        ----------
        cmap : str, optional
            Colormap to use, by default "RdBu_r".
        **kwargs : Any
            Additional arguments passed to sns.heatmap.

        Returns
        -------
        matplotlib.axes.Axes
            The axes object.

        Examples
        --------
        >>> # Assuming 'plot' is a DiurnalErrorPlot instance
        >>> ax = plot.plot(cmap="viridis")
        """
        if self.aggregated is None:
            raise ValueError("Aggregated data not found. Call _calculate_metric first.")

        # Compute the aggregated data for plotting
        data_to_plot = self.aggregated
        if hasattr(data_to_plot.data, "chunks"):
            data_to_plot = data_to_plot.compute()

        # Convert to DataFrame for Seaborn
        plot_df = data_to_plot.to_pandas()

        sns.heatmap(
            plot_df,
            ax=self.ax,
            cmap=cmap,
            center=0 if self.metric == "bias" else None,
            **kwargs,
        )

        self.ax.set_xlabel("Hour of Day")
        self.ax.set_ylabel(self.second_label)
        self.ax.set_title(f"Diurnal {self.metric.capitalize()}")

        return self.ax

    def hvplot(self, cmap: str = "RdBu_r", **kwargs: Any) -> hv.Element:
        """
        Generate the diurnal error heatmap (Track B: Interactive).

        Parameters
        ----------
        cmap : str, optional
            Colormap to use, by default "RdBu_r".
        **kwargs : Any
            Additional arguments passed to hvplot.heatmap.

        Returns
        -------
        holoviews.Element
            The interactive HoloViews object.

        Examples
        --------
        >>> # Assuming 'plot' is a DiurnalErrorPlot instance
        >>> interactive_plot = plot.hvplot()
        """
        import hvplot.xarray  # noqa: F401

        if self.aggregated is None:
            raise ValueError("Aggregated data not found. Call _calculate_metric first.")

        # Track B: Interactive
        return self.aggregated.hvplot.heatmap(
            x="hour",
            y="second_val",
            C=self.aggregated.name,
            cmap=cmap,
            title=f"Diurnal {self.metric.capitalize()}",
            xlabel="Hour of Day",
            ylabel=self.second_label,
            rasterize=True,
            **kwargs,
        )
