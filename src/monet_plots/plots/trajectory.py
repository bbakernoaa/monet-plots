from __future__ import annotations

import typing as t

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from ..plot_utils import _update_history, normalize_data
from .base import BasePlot
from .spatial import SpatialTrack
from .timeseries import TimeSeriesPlot


class TrajectoryPlot(BasePlot):
    """Plot a trajectory on a map and a timeseries of a variable."""

    def __init__(
        self,
        longitude: t.Any,
        latitude: t.Any,
        data: t.Any,
        time: t.Any,
        ts_data: t.Any,
        fig: plt.Figure | None = None,
        ax: list[plt.Axes] | None = None,
        **kwargs: t.Any,
    ) -> None:
        """Initialize the trajectory plot.

        This plot combines a spatial map showing the path (track) and a
        timeseries plot of a variable along that path.

        Parameters
        ----------
        longitude : ArrayLike
            Longitude values for the spatial track.
        latitude : ArrayLike
            Latitude values for the spatial track.
        data : ArrayLike
            Data values used for coloring the spatial track.
        time : ArrayLike or pd.DataFrame or xr.DataArray
            Time values for the timeseries, or a DataFrame/DataArray containing
            the timeseries data.
        ts_data : ArrayLike or str
            Data for the timeseries, or the column/variable name if `time`
            is a DataFrame or DataArray.
        fig : plt.Figure, optional
            An existing Figure object. If None, one is created.
        ax : list of plt.Axes, optional
            A list of two axes [map_ax, ts_ax]. If None, they are created.
        **kwargs : Any
            Additional keyword arguments passed to :class:`BasePlot`.
        """
        if fig is None and ax is None:
            fig = plt.figure()

        super().__init__(fig=fig, ax=ax, **kwargs)

        # Normalize data to support lazy objects
        self.longitude = normalize_data(longitude)
        self.latitude = normalize_data(latitude)
        self.data = normalize_data(data)
        self.time = normalize_data(time)
        self.ts_data = ts_data

        _update_history(self.data, "Initialized TrajectoryPlot (Map)")
        _update_history(self.time, "Initialized TrajectoryPlot (Time)")

    def plot(self, **kwargs: t.Any) -> list[plt.Axes]:
        """Plot the trajectory and timeseries.

        Parameters
        ----------
        **kwargs : Any
            Keyword arguments passed to the plot methods.
            - `spatial_track_kwargs`: Dict of kwargs for :class:`SpatialTrack`.
            - `timeseries_kwargs`: Dict of kwargs for :class:`TimeSeriesPlot`.
            - `projection`: Cartopy projection for the map, defaults to PlateCarree.
            - `figsize`: Figure size, defaults to (12, 6).

        Returns
        -------
        list of plt.Axes
            The [map_ax, ts_ax] created.
        """
        if self.fig is None:
            self.fig = plt.figure(figsize=kwargs.get("figsize", (12, 6)))

        # Ensure constrained_layout to help with alignment
        self.fig.set_constrained_layout(True)

        # Create axes if not provided
        if self.ax is None:
            gs = self.fig.add_gridspec(2, 1, height_ratios=[3, 1])

            # Spatial track plot
            import cartopy.crs as ccrs

            proj = kwargs.get("projection", ccrs.PlateCarree())
            ax0 = self.fig.add_subplot(gs[0, 0], projection=proj)

            # Set adjustable to 'datalim' to allow the map to fill the horizontal
            # space while maintaining equal aspect ratio by expanding the limits.
            ax0.set_adjustable("datalim")

            # Timeseries plot
            ax1 = self.fig.add_subplot(gs[1, 0])
            self.ax = [ax0, ax1]
        else:
            ax0, ax1 = self.ax

        # Create an xarray.DataArray for the trajectory data lazily
        if isinstance(self.data, xr.DataArray):
            track_da = self.data.copy()
            # If coordinates are separate, add them
            if not isinstance(self.longitude, (xr.DataArray, xr.Dataset)) or (
                "lon" not in track_da.coords
            ):
                track_da.coords["lon"] = self.longitude
            if not isinstance(self.latitude, (xr.DataArray, xr.Dataset)) or (
                "lat" not in track_da.coords
            ):
                track_da.coords["lat"] = self.latitude
        else:
            # Fallback for pandas/numpy
            lon = np.asarray(self.longitude).squeeze()
            lat = np.asarray(self.latitude).squeeze()
            values = np.asarray(self.data).squeeze()
            time_dim = np.arange(len(lon))
            coords = {"time": time_dim, "lon": ("time", lon), "lat": ("time", lat)}
            track_da = xr.DataArray(
                values, dims=["time"], coords=coords, name="track_data"
            )

        _update_history(track_da, "Plotted with monet-plots.TrajectoryPlot (Map)")

        # Pass the DataArray to SpatialTrack
        plot_kwargs = kwargs.get("spatial_track_kwargs", {})
        spatial_track = SpatialTrack(data=track_da, ax=ax0, fig=self.fig)
        spatial_track.plot(**plot_kwargs)

        # Timeseries plot
        timeseries_kwargs = kwargs.get("timeseries_kwargs", {}).copy()

        # Handle time series data lazily
        if isinstance(self.ts_data, str) and isinstance(
            self.time, (xr.DataArray, xr.Dataset, pd.DataFrame)
        ):
            _update_history(self.time, "Plotted with monet-plots.TrajectoryPlot (TS)")
            timeseries = TimeSeriesPlot(
                df=self.time, y=self.ts_data, ax=ax1, fig=self.fig
            )
        else:
            # Fallback for arrays/unnamed data
            t_vals = np.asarray(self.time).squeeze()
            y_vals = np.asarray(self.ts_data).squeeze()
            ts_df = pd.DataFrame({"time": t_vals, "value": y_vals})
            timeseries = TimeSeriesPlot(
                df=ts_df, x="time", y="value", ax=ax1, fig=self.fig
            )

        timeseries.plot(**timeseries_kwargs)

        return self.ax

    def hvplot(self, **kwargs: t.Any) -> t.Any:
        """Interactive visualization of the trajectory.

        Parameters
        ----------
        **kwargs : Any
            Keyword arguments passed to hvplot methods.
            - `spatial_track_kwargs`: Dict of kwargs for the spatial hvplot.
            - `timeseries_kwargs`: Dict of kwargs for the timeseries hvplot.

        Returns
        -------
        holoviews.Layout
            A vertical layout of the interactive spatial track and timeseries.
        """
        import hvplot.pandas  # noqa: F401
        import hvplot.xarray  # noqa: F401

        spatial_kwargs = kwargs.get("spatial_track_kwargs", {}).copy()
        ts_kwargs = kwargs.get("timeseries_kwargs", {}).copy()

        # Prepare spatial data
        if isinstance(self.data, xr.DataArray):
            track_da = self.data.copy()
            if not isinstance(self.longitude, (xr.DataArray, xr.Dataset)) or (
                "lon" not in track_da.coords
            ):
                track_da.coords["lon"] = self.longitude
            if not isinstance(self.latitude, (xr.DataArray, xr.Dataset)) or (
                "lat" not in track_da.coords
            ):
                track_da.coords["lat"] = self.latitude
        else:
            lon = np.asarray(self.longitude).squeeze()
            lat = np.asarray(self.latitude).squeeze()
            values = np.asarray(self.data).squeeze()
            time_dim = np.arange(len(lon))
            coords = {"time": time_dim, "lon": ("time", lon), "lat": ("time", lat)}
            track_da = xr.DataArray(
                values, dims=["time"], coords=coords, name="track_data"
            )

        spatial_kwargs.setdefault("geo", True)
        spatial_kwargs.setdefault("coastline", True)
        spatial_kwargs.setdefault("x", "lon")
        spatial_kwargs.setdefault("y", "lat")
        spatial_kwargs.setdefault("c", track_da.name or "value")

        spatial_p = track_da.hvplot.scatter(**spatial_kwargs)

        # Prepare timeseries data
        if isinstance(self.ts_data, str) and isinstance(
            self.time, (xr.DataArray, xr.Dataset, pd.DataFrame)
        ):
            # If it's a DataFrame/Dataset, ts_data is expected to be the variable name
            ts_p = self.time.hvplot(y=self.ts_data, **ts_kwargs)
        else:
            t_vals = np.asarray(self.time).squeeze()
            y_vals = np.asarray(self.ts_data).squeeze()
            ts_df = pd.DataFrame({"time": t_vals, "value": y_vals})
            ts_p = ts_df.hvplot(x="time", y="value", **ts_kwargs)

        return (spatial_p + ts_p).cols(1)
