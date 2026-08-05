# src/monet_plots/plots/fingerprint.py
"""Fingerprint plot for visualizing temporal patterns."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
import seaborn as sns
import xarray as xr

from ..plot_utils import _update_history, compute, is_lazy, normalize_data
from .base import BasePlot

if TYPE_CHECKING:
    import holoviews as hv
    import matplotlib.axes
    import matplotlib.figure


class FingerprintPlot(BasePlot):
    """Fingerprint plot.

    Displays a variable as a heatmap across two different temporal scales,
    such as hour of day vs. day of year, to reveal periodic patterns.

    This class supports native Xarray and Dask objects for lazy evaluation
    and provenance tracking.

    Attributes
    ----------
    data : Union[xr.Dataset, xr.DataArray, pd.DataFrame]
        The input data for the plot.
    val_col : str
        Column/variable name of the value to plot.
    time_col : str
        Dimension/column name for timestamp.
    x_scale : str
        Temporal scale for the x-axis ('hour', 'month', 'dayofweek', etc.).
    y_scale : str
        Temporal scale for the y-axis ('dayofyear', 'year', 'week', etc.).
    aggregated : xr.DataArray
        The calculated 2D aggregated mean values for the heatmap.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from monet_plots.plots import FingerprintPlot
    >>> dates = pd.date_range("2023-01-01", periods=1000, freq="h")
    >>> df = pd.DataFrame({"time": dates, "val": np.random.rand(1000)})
    >>> plot = FingerprintPlot(df, val_col="val")
    >>> ax = plot.plot()
    """

    def __init__(
        self,
        data: Any,
        val_col: str,
        *,
        time_col: str = "time",
        x_scale: str = "hour",
        y_scale: str = "dayofyear",
        fig: matplotlib.figure.Figure | None = None,
        ax: matplotlib.axes.Axes | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize Fingerprint Plot.

        Parameters
        ----------
        data : Any
            Input data. Can be a pandas DataFrame, xarray DataArray,
            xarray Dataset, or dask-backed object.
        val_col : str
            Column name of the value to plot.
        time_col : str, optional
            Dimension/column name for timestamp, by default "time".
        x_scale : str, optional
            Temporal scale for the x-axis ('hour', 'month', 'dayofweek', etc.),
            by default "hour".
        y_scale : str, optional
            Temporal scale for the y-axis ('dayofyear', 'year', 'week', etc.),
            by default "dayofyear".
        fig : matplotlib.figure.Figure, optional
            Existing figure object, by default None.
        ax : matplotlib.axes.Axes, optional
            Existing axes object, by default None.
        **kwargs : Any
            Additional arguments passed to BasePlot.
        """
        super().__init__(fig=fig, ax=ax, **kwargs)

        # Normalize data to Xarray if possible, preserving lazy evaluation
        self.data = normalize_data(data)
        self.val_col = val_col
        self.time_col = time_col
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.aggregated: xr.DataArray | None = None

        # Prepare and calculate the 2D aggregation matrix
        self._calculate_aggregation()

    def _extract_scale_xr(self, ds: xr.Dataset, scale: str) -> xr.DataArray:
        """Extract temporal scale coordinate from an xarray dataset.

        Parameters
        ----------
        ds : xr.Dataset
            The input dataset.
        scale : str
            The temporal scale to extract ('hour', 'month', 'dayofweek', etc.).

        Returns
        -------
        xr.DataArray
            The extracted scale as a coordinate DataArray.

        Raises
        ------
        ValueError
            If the requested scale is unknown and not found in dataset.
        """
        time_coord = ds[self.time_col]
        t = time_coord.dt

        if scale == "hour":
            return t.hour
        elif scale == "month":
            return t.month
        elif scale == "dayofweek":
            return t.dayofweek
        elif scale == "dayofyear":
            return t.dayofyear
        elif scale == "week":
            return t.isocalendar().week
        elif scale == "year":
            return t.year
        elif scale == "date":
            return time_coord.dt.floor("D")
        else:
            if scale in ds.coords:
                return ds.coords[scale]
            elif scale in ds.data_vars:
                return ds[scale]
            else:
                raise ValueError(f"Unknown temporal scale: {scale}")

    def _extract_scale_pd(self, df: pd.DataFrame, scale: str) -> pd.Series | Any:
        """Extract temporal scale from a pandas DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame.
        scale : str
            The temporal scale to extract ('hour', 'month', 'dayofweek', etc.).

        Returns
        -------
        pd.Series | Any
            The extracted scale.

        Raises
        ------
        ValueError
            If the requested scale is unknown and not found in DataFrame.
        """
        t = df[self.time_col].dt

        if scale == "hour":
            return t.hour
        elif scale == "month":
            return t.month
        elif scale == "dayofweek":
            return t.dayofweek
        elif scale == "dayofyear":
            return t.dayofyear
        elif scale == "week":
            return t.isocalendar().week
        elif scale == "year":
            return t.year
        elif scale == "date":
            return t.date
        else:
            if scale in df.columns:
                return df[scale]
            else:
                raise ValueError(f"Unknown temporal scale: {scale}")

    def _calculate_aggregation(self) -> None:
        """Calculate the 2D aggregated mean for the fingerprint heatmap.

        This method identifies the appropriate backend, extracts the temporal
        scales, and aggregates the values into a 2D matrix (y_val, x_val).
        It preserves lazy evaluation for Dask and Cubed objects.
        """
        ds = self.data
        if isinstance(ds, xr.DataArray):
            ds = ds.to_dataset()

        if isinstance(ds, xr.Dataset):
            val = ds[self.val_col]

            # Extract coordinates using the xarray helper
            x_coord = self._extract_scale_xr(ds, self.x_scale)
            y_coord = self._extract_scale_xr(ds, self.y_scale)

            # Assign extracted scales as coordinate arrays
            val = val.assign_coords(x_val=x_coord, y_val=y_coord)

            # Vectorized multi-dimensional grouping
            try:
                self.aggregated = val.groupby(["y_val", "x_val"]).mean(
                    dim=self.time_col, keep_attrs=True
                )
                if "y_val" in self.aggregated.dims and "x_val" in self.aggregated.dims:
                    self.aggregated = self.aggregated.transpose("y_val", "x_val")
            except (TypeError, ValueError, AttributeError, KeyError):
                # Eager fallback if complex multi-dimensional grouping fails
                df = val.to_dataframe(name=val.name).reset_index()
                pivot = df.pivot_table(
                    index="y_val", columns="x_val", values=val.name, aggfunc="mean"
                )
                self.aggregated = xr.DataArray(
                    pivot.values,
                    coords={
                        "y_val": pivot.index.values,
                        "x_val": pivot.columns.values,
                    },
                    dims=["y_val", "x_val"],
                    name=val.name,
                )

            self.aggregated = _update_history(
                self.aggregated, f"Calculated fingerprint for {self.val_col}"
            )
        else:
            # Fallback for Pandas DataFrame (backward compatibility)
            df = self.data.copy()
            df[self.time_col] = pd.to_datetime(df[self.time_col])

            df["x_val"] = self._extract_scale_pd(df, self.x_scale)
            df["y_val"] = self._extract_scale_pd(df, self.y_scale)

            pivot = df.pivot_table(
                index="y_val", columns="x_val", values=self.val_col, aggfunc="mean"
            )

            self.aggregated = xr.DataArray(
                pivot.values,
                coords={
                    "y_val": pivot.index.values,
                    "x_val": pivot.columns.values,
                },
                dims=["y_val", "x_val"],
                name=self.val_col,
            )
            self.aggregated = _update_history(
                self.aggregated, f"Calculated fingerprint for {self.val_col}"
            )

    def plot(self, cmap: str = "viridis", **kwargs: Any) -> matplotlib.axes.Axes:
        """Generate the fingerprint heatmap (Track A: Static).

        Parameters
        ----------
        cmap : str, optional
            Colormap to use, by default "viridis".
        **kwargs : Any
            Additional arguments passed to sns.heatmap.

        Returns
        -------
        matplotlib.axes.Axes
            The axes object.

        Examples
        --------
        >>> # Assuming 'plot' is a FingerprintPlot instance
        >>> ax = plot.plot(cmap="plasma")
        """
        if self.aggregated is None:
            raise ValueError(
                "Aggregated data not found. Call _calculate_aggregation first."
            )

        # Compute the aggregated data if it's lazy before plotting
        data_to_plot = self.aggregated
        if is_lazy(data_to_plot):
            data_to_plot = compute(data_to_plot)

        # Convert to DataFrame for Seaborn heatmap
        plot_df = data_to_plot.to_pandas()

        sns.heatmap(plot_df, ax=self.ax, cmap=cmap, **kwargs)

        self.ax.set_xlabel(self.x_scale.capitalize())
        self.ax.set_ylabel(self.y_scale.capitalize())
        self.ax.set_title(f"Fingerprint: {self.val_col}")

        return self.ax

    def hvplot(self, cmap: str = "viridis", **kwargs: Any) -> hv.Element:
        """Generate the fingerprint heatmap (Track B: Interactive).

        Parameters
        ----------
        cmap : str, optional
            Colormap to use, by default "viridis".
        **kwargs : Any
            Additional arguments passed to hvplot.heatmap.

        Returns
        -------
        holoviews.Element
            The interactive HoloViews object.

        Examples
        --------
        >>> # Assuming 'plot' is a FingerprintPlot instance
        >>> interactive_plot = plot.hvplot()
        """
        try:
            import hvplot.xarray  # noqa: F401
        except ImportError:
            raise ImportError(
                "hvplot is required for interactive plotting. Install it with 'pip install hvplot'."
            )

        if self.aggregated is None:
            raise ValueError(
                "Aggregated data not found. Call _calculate_aggregation first."
            )

        # Return high-performance interactive rasterized heatmap
        return self.aggregated.hvplot.heatmap(
            x="x_val",
            y="y_val",
            C=self.aggregated.name,
            cmap=cmap,
            title=f"Fingerprint: {self.val_col}",
            xlabel=self.x_scale.capitalize(),
            ylabel=self.y_scale.capitalize(),
            rasterize=True,
            **kwargs,
        )
