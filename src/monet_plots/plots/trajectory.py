from __future__ import annotations

import typing as t

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

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
        *args: t.Any,
        **kwargs: t.Any,
    ) -> None:
        """
        Initialize the trajectory plot.
        Args:
            longitude: Longitude values for the spatial track.
            latitude: Latitude values for the spatial track.
            data: Data to use for coloring the track.
            time: Time values for the timeseries.
            ts_data: Data for the timeseries.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.longitude = longitude
        self.latitude = latitude
        self.data = data
        self.time = time
        self.ts_data = ts_data

    def plot(self, **kwargs: t.Any) -> None:
        """Plot the trajectory and timeseries.

        Args:
            **kwargs: Keyword arguments passed to the plot methods.
        """
        if self.fig is None:
            self.fig = plt.figure(figsize=kwargs.get("figsize", (10, 8)))

        gs = self.fig.add_gridspec(2, 1, height_ratios=[3, 1])

        # Spatial track plot
        ax0 = self.fig.add_subplot(gs[0, 0], projection=kwargs.get("projection"))

        # Create an xarray.DataArray for the trajectory data
        lon = np.asarray(self.longitude)
        lat = np.asarray(self.latitude)
        values = np.asarray(self.data)
        time_dim = np.arange(len(lon))
        coords = {"time": time_dim, "lon": ("time", lon), "lat": ("time", lat)}
        track_da = xr.DataArray(values, dims=["time"], coords=coords, name="track_data")

        # Pass the DataArray to SpatialTrack
        plot_kwargs = kwargs.get("spatial_track_kwargs", {})
        spatial_track = SpatialTrack(data=track_da, ax=ax0)
        spatial_track.plot(**plot_kwargs)

        # Timeseries plot
        ax1 = self.fig.add_subplot(gs[1, 0])
        timeseries = TimeSeriesPlot(data=self.time, y=self.ts_data, ax=ax1)
        timeseries.plot(**kwargs.get("timeseries_kwargs", {}))

        self.ax = [ax0, ax1]
