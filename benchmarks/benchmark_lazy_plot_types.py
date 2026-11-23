"""Benchmark one static plot with Dask-backed inputs in a fresh Linux process.

Run with PYTHONPATH=src and MPLBACKEND=Agg. Set CARTOPY_DATA_DIR to an
existing Cartopy data cache if your environment requires one. The JSON
output separates Dask scheduling during construction, plotting, and drawing.
Peak RSS is cumulative within the process. Each task count represents one
optimized scheduler run with two Dask worker threads.
"""

import argparse
import json
import resource
from os import environ
from pathlib import Path
from time import perf_counter

import cartopy
import dask
import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from dask.callbacks import Callback
from matplotlib import pyplot as plt

from monet_plots.plots.conditional_bias import ConditionalBiasPlot
from monet_plots.plots.curtain import CurtainPlot
from monet_plots.plots.diurnal_error import DiurnalErrorPlot
from monet_plots.plots.facet_grid import FacetGridPlot
from monet_plots.plots.radar import RadarPlot
from monet_plots.plots.reliability_diagram import ReliabilityDiagramPlot
from monet_plots.plots.ridgeline import RidgelinePlot
from monet_plots.plots.scatter import ScatterPlot
from monet_plots.plots.soccer import SoccerPlot
from monet_plots.plots.spatial import SpatialTrack
from monet_plots.plots.spatial_contour import SpatialContourPlot
from monet_plots.plots.spatial_imshow import SpatialImshowPlot
from monet_plots.plots.timeseries import TimeSeriesErrorBarPlot, TimeSeriesStatsPlot
from monet_plots.plots.trajectory import TrajectoryPlot
from monet_plots.plots.windrose import Windrose


class TaskCounts(Callback):
    """Count optimized scheduler invocations and executed tasks."""

    def __init__(self):
        super().__init__()
        self.tasks = []

    def _start(self, graph):
        self.tasks.append(0)

    def _pretask(self, key, graph, state):
        self.tasks[-1] += 1


def observation_pairs(n, chunks):
    rng = da.random.RandomState(42)
    obs = 1 + rng.random_sample(n, chunks=chunks)
    noise = rng.standard_normal(n, chunks=chunks)
    return xr.Dataset({"obs": ("sample", obs), "model": ("sample", obs + 0.2 * noise)})


def geographic_field(facets=False):
    rng = da.random.RandomState(42)
    lat = np.linspace(25, 50, 384)
    lon = np.linspace(-120, -72, 512)
    if facets:
        shape, chunks = (4, lat.size, lon.size), (1, 96, 128)
        dims = ("time", "lat", "lon")
        coords = {"time": np.arange(shape[0]), "lat": lat, "lon": lon}
    else:
        shape, chunks = (lat.size, lon.size), (96, 128)
        dims = ("lat", "lon")
        coords = {"lat": lat, "lon": lon}
    noise = rng.random_sample(shape, chunks=chunks)
    field = np.cos(lat[:, None] / 5) + np.sin(lon / 5) + 0.1 * noise
    return xr.DataArray(field, dims=dims, coords=coords, name="Field")


def build_case(case):
    """Return constructor, renderer, and description of workload."""
    if case in {"reliability-xarray", "reliability-dataframe"}:
        n, chunks = 1_048_576, 32_768
        rng = da.random.RandomState(42)
        forecast = rng.random_sample(n, chunks=chunks)
        event = (rng.random_sample(n, chunks=chunks) < forecast).astype("uint8")
        data = xr.Dataset(
            {
                "forecast": ("sample", forecast),
                "event": ("sample", event),
                "model": ("sample", da.arange(n, chunks=chunks) % 4),
            }
        )
        if case == "reliability-dataframe":
            data = data.to_dask_dataframe()
        return (
            ReliabilityDiagramPlot,
            lambda p: p.plot(
                data,
                forecasts_col="forecast",
                observations_col="event",
                label_col="model",
                n_bins=10,
                show_hist=True,
            ),
            n,
        )
    if case == "timeseries-errorbars":
        shape, chunks = (168, 2_048), (24, 256)
        rng = da.random.RandomState(42)
        obs = rng.random_sample(shape, chunks=chunks)
        model = obs + 0.2 * rng.standard_normal(shape, chunks=chunks)
        dims = ("time", "station")
        data = xr.Dataset(
            {"obs": (dims, obs), "model": (dims, model), "model2": (dims, model + 0.1)},
            coords={"time": pd.date_range("2020", periods=shape[0], freq="h")},
        )
        return (
            lambda: TimeSeriesErrorBarPlot(data, y=["obs", "model", "model2"]),
            lambda p: p.plot(),
            obs.size,
        )
    if case == "timeseries-stats":
        n = 24 * 365 * 3
        base = observation_pairs(n, 24 * 30).rename(sample="time")
        data = base.assign_coords(time=pd.date_range("2020", periods=n, freq="h"))
        data = data.assign(model2=data.model + 0.1, model3=data.model - 0.2)
        models = ["model", "model2", "model3"]
        return (
            lambda: TimeSeriesStatsPlot(data, "obs", models),
            lambda p: p.plot(stat="rmse", freq="D"),
            n,
        )
    if case in {"scatter", "soccer", "radar"} or case.startswith("conditional"):
        n = 131_072 if case == "scatter" else 524_288
        data = observation_pairs(n, 16_384)
        if case == "scatter":
            return lambda: ScatterPlot(data, "obs", "model"), lambda p: p.plot(s=2), n
        if case == "soccer":
            return (
                lambda: SoccerPlot(data, obs_col="obs", mod_col="model"),
                lambda p: p.plot(s=2),
                n,
            )
        if case == "radar":

            def radar():
                models = data.assign(model2=data.model + 0.1)
                return RadarPlot(models, obs_col="obs", mod_cols=["model", "model2"])

            return radar, lambda p: p.plot(), n
        plot_options = {
            "n_bins": np.linspace(1, 2, 13) if case != "conditional" else 12
        }
        if case == "conditional-chunked-labels":
            region = da.arange(n, chunks=16_384) // (n // 4)
            data = data.assign_coords(region=("sample", region))
            plot_options["label_col"] = "region"
        return (
            lambda: ConditionalBiasPlot(data=data),
            lambda p: p.plot(var1="obs", var2="model", **plot_options),
            n,
        )

    if case.startswith("diurnal"):
        n = 24 * 365 * 12
        data = observation_pairs(n, 24 * 28).rename(sample="time")
        data = data.assign_coords(time=pd.date_range("2010", periods=n, freq="h"))
        second_dim = "month"
        if case.endswith("chunked-labels"):
            regions = da.arange(n, chunks=24 * 28) // 24 % 4
            data = data.assign_coords(region=("time", regions))
            second_dim = "region"
        return (
            lambda: DiurnalErrorPlot(data, "obs", "model", second_dim=second_dim),
            lambda p: p.plot(),
            n,
        )

    if case.startswith("windrose"):
        n = 1_048_576
        rng = da.random.RandomState(42)
        directions = xr.DataArray(360 * rng.random_sample(n, chunks=32_768))
        speeds = xr.DataArray(30 * rng.random_sample(n, chunks=32_768))
        bins = np.linspace(0, 30, 6) if case.endswith("fixed") else 5
        return (
            lambda: Windrose(wd=directions, ws=speeds),
            lambda p: p.plot(rose_bins=bins),
            n,
        )

    if case.startswith("ridgeline"):
        n = 8 * 4_096
        rng = da.random.RandomState(42)
        data = xr.DataArray(
            rng.standard_normal((8, 4_096), chunks=(1, 1024)),
            dims=("group", "sample"),
            coords={"group": np.arange(8)},
        )
        x_range = (-5, 5) if case.endswith("fixed") else None
        return (
            lambda: RidgelinePlot(data, group_dim="group", x_range=x_range),
            lambda p: p.plot(gradient=False),
            n,
        )

    if case.startswith("track") or case == "trajectory":
        n = 65_536
        rng = da.random.RandomState(42)
        data = xr.DataArray(
            rng.random_sample(n, chunks=8_192),
            dims="sample",
            coords={
                "lon": ("sample", da.linspace(-120, -80, n, chunks=8_192)),
                "lat": ("sample", da.linspace(30, 45, n, chunks=8_192)),
            },
        )
        if case == "trajectory":
            times = pd.date_range("2020", periods=n, freq="min")
            return (
                lambda: TrajectoryPlot(data.lon, data.lat, data, times, data),
                lambda p: p.plot(spatial_track_kwargs={"coastlines": False, "s": 2}),
                n,
            )
        extent = {"extent": [-122, -78, 28, 47]} if case.endswith("fixed") else {}
        return (
            lambda: SpatialTrack(data, coastlines=False),
            lambda p: p.plot(coastlines=False, s=2, **extent),
            n,
        )

    if case == "facet-line":
        shape, chunks = (4, 8192), (1, 2048)
        rng = da.random.RandomState(42)
        values = rng.random_sample(shape, chunks=chunks)
        data = xr.Dataset(
            {"signal": (("facet", "sample"), values)},
            coords={"facet": np.arange(shape[0]), "sample": np.arange(shape[1])},
        )
        return (
            lambda: FacetGridPlot(data, col="facet", col_wrap=2),
            lambda p: p.plot(plt.plot, "sample", "signal"),
            values.size,
        )

    field = geographic_field(facets=case.startswith("facet"))
    if case.startswith("facet"):
        plot_type = (
            SpatialContourPlot if case.endswith("contour") else SpatialImshowPlot
        )
        return (
            lambda: SpatialImshowPlot(field, col="time", col_wrap=2),
            lambda p: p.map_monet(plot_type, coastlines=False, add_colorbar=True),
            field.size,
        )
    if case == "imshow":
        return (
            lambda: SpatialImshowPlot(field, coastlines=False),
            lambda p: p.plot(),
            field.size,
        )
    if case.startswith("contour"):
        kwargs = {"levels": np.linspace(-2, 2.1, 10)} if case.endswith("fixed") else {}
        return (
            lambda: SpatialContourPlot(field, coastlines=False),
            lambda p: p.plot(**kwargs),
            field.size,
        )
    kind = "contourf" if case.endswith("contour") else "pcolormesh"
    return lambda: CurtainPlot(field), lambda p: p.plot(kind=kind), field.size


CASES = (
    "reliability-xarray",
    "reliability-dataframe",
    "timeseries-errorbars",
    "timeseries-stats",
    "scatter",
    "soccer",
    "radar",
    "conditional",
    "conditional-fixed",
    "conditional-chunked-labels",
    "diurnal",
    "diurnal-chunked-labels",
    "windrose",
    "windrose-fixed",
    "ridgeline",
    "ridgeline-fixed",
    "track",
    "track-fixed",
    "trajectory",
    "imshow",
    "contour",
    "contour-fixed",
    "facet-imshow",
    "facet-contour",
    "facet-line",
    "curtain-mesh",
    "curtain-contour",
)


def measure(function):
    counts = TaskCounts()
    start = perf_counter()
    with counts:
        output = function()
    return output, {
        "seconds": round(perf_counter() - start, 3),
        "tasks_by_run": counts.tasks,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def main():
    plt.switch_backend("Agg")
    if cache := environ.get("CARTOPY_DATA_DIR"):
        cartopy.config["data_dir"] = Path(cache)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=CASES, nargs="?")
    parser.add_argument("--list", action="store_true", help="List available workloads")
    args = parser.parse_args()
    if args.list:
        print("\n".join(CASES))
        return
    if args.case is None:
        parser.error("choose a workload or use --list")
    construct, plot, size = build_case(args.case)
    baseline = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    with dask.config.set(scheduler="threads", num_workers=2):
        plotter, init = measure(construct)
        _, render = measure(lambda: plot(plotter))
        _, draw = measure(plotter.fig.canvas.draw)
    print(
        json.dumps(
            {
                "case": args.case,
                "samples": size,
                "baseline_rss_kib": baseline,
                "init": init,
                "plot": render,
                "draw": draw,
            }
        )
    )
    plt.close("all")


if __name__ == "__main__":
    main()
