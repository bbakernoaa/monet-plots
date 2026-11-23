"""Measure Dask task executions, render time, and Linux peak RSS in a fresh process."""

import argparse
import json
import resource
from types import SimpleNamespace
from time import perf_counter

import dask
import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from dask.callbacks import Callback
from matplotlib import pyplot as plt

from monet_plots.plots.spatial_bias_scatter import SpatialBiasScatterPlot
from monet_plots.plots.timeseries import TimeSeriesPlot
from monet_plots.plots.wind_barbs import WindBarbsPlot
from monet_plots.plots.wind_quiver import WindQuiverPlot


class TaskCounts(Callback):
    """Count optimized scheduler runs and executed tasks."""

    runs = 0
    tasks = 0

    def _start(self, graph):
        self.runs += 1

    def _pretask(self, key, graph, state):
        self.tasks += 1


def spatial(chunked_coordinates=False):
    rng = da.random.RandomState(42)
    shape = (512, 768)
    chunks = (128, 128)
    obs = rng.random_sample(shape, chunks=chunks)
    data = xr.Dataset(
        {
            "obs": (("row", "col"), obs),
            "model": (("row", "col"), obs + rng.standard_normal(shape, chunks=chunks)),
        },
        coords={
            "lat": ("row", np.linspace(25, 50, shape[0])),
            "lon": ("col", np.linspace(-125, -65, shape[1])),
        },
    )
    if chunked_coordinates:
        data = data.chunk(dict(zip(("row", "col"), chunks, strict=True)))
    return SpatialBiasScatterPlot(
        data, "obs", "model", coastlines=False, style="journal"
    )


def timeseries():
    rng = da.random.RandomState(42)
    shape = (168, 256, 256)
    values = rng.standard_normal(shape, chunks=(24, 64, 64))
    data = xr.DataArray(
        values,
        name="temperature",
        dims=("time", "lat", "lon"),
        coords={"time": pd.date_range("2025-01-01", periods=shape[0], freq="h")},
    )
    return TimeSeriesPlot(data, style="journal")


def wind(plot_type):
    rng = da.random.RandomState(42)
    shape = (1024, 1536)
    chunks = (32, 32)
    ws = rng.uniform(1, 30, size=shape, chunks=chunks)
    direction = rng.uniform(0, 360, size=shape, chunks=chunks)
    lat = np.broadcast_to(np.linspace(25, 50, shape[0])[:, None], shape)
    lon = np.broadcast_to(np.linspace(-125, -65, shape[1]), shape)
    grid = SimpleNamespace(variables={"LAT": lat[None, None], "LON": lon[None, None]})
    cls = WindBarbsPlot if plot_type == "wind-barbs" else WindQuiverPlot
    return cls(ws, direction, grid, coastlines=False, style="journal")


def main():
    plt.switch_backend("Agg")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "plot",
        choices=[
            "spatial",
            "spatial-chunked-coordinates",
            "timeseries",
            "wind-barbs",
            "wind-quiver",
        ],
    )
    plot_type = parser.parse_args().plot
    is_wind = plot_type.startswith("wind-")
    plot = (
        timeseries()
        if plot_type == "timeseries"
        else None
        if is_wind
        else spatial(plot_type.endswith("coordinates"))
    )
    counts = TaskCounts()
    start = perf_counter()
    with dask.config.set(scheduler="threads", num_workers=2), counts:
        if is_wind:
            # Include construction because older wind plots load the input here.
            plot = wind(plot_type)
        plot.plot()
        plot.fig.canvas.draw()
    print(
        json.dumps(
            {
                "plot": plot_type,
                "seconds": round(perf_counter() - start, 3),
                "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "scheduler_runs": counts.runs,
                "executed_tasks": counts.tasks,
            }
        )
    )
    plot.close()


if __name__ == "__main__":
    main()
