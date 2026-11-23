"""Measure Dask task executions, render time, and Linux peak RSS in a fresh process."""

import argparse
import json
import resource
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


def main():
    plt.switch_backend("Agg")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "plot", choices=["spatial", "spatial-chunked-coordinates", "timeseries"]
    )
    plot_type = parser.parse_args().plot
    plot = (
        timeseries()
        if plot_type == "timeseries"
        else spatial(plot_type.endswith("coordinates"))
    )
    counts = TaskCounts()
    start = perf_counter()
    with dask.config.set(scheduler="threads", num_workers=2), counts:
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
