"""Benchmark first headless renders of MONET Plots static workloads.

Run each workload in a fresh Python process. Initialize MONET Plots and Agg,
then time fixture preparation, plot construction, and the first canvas draws.
Gallery cases run their example scripts. An example's save/show call draws its
figure without encoding an output file. Peak RSS includes process imports and
native allocations. The inputs vary by plot type, so compare like workloads.

For an offline run, pass --cartopy-data with cached Natural Earth 50m coastlines
and admin_1_states_provinces_lakes boundaries.
"""

from __future__ import annotations

import argparse
import ast
import csv
import importlib
import json
import multiprocessing
import os
import resource
import runpy
import statistics
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"

GALLERY = {
    "BivariatePolarPlot": "polar",
    "BrierScoreDecompositionPlot": "brier_decomposition",
    "ConditionalBiasPlot": "conditional_bias",
    "ConditionalQuantilePlot": "conditional_quantile",
    "CurtainPlot": "curtain",
    "DiurnalErrorPlot": "diurnal_error",
    "FacetGridPlot": "facet_grid",
    "FingerprintPlot": "fingerprint",
    "KDEPlot": "kde",
    "Meteogram": "meteogram",
    "PerformanceDiagramPlot": "performance_diagram",
    "ProfilePlot": "profile",
    "RankHistogramPlot": "rank_histogram",
    "RelativeEconomicValuePlot": "rev",
    "ReliabilityDiagramPlot": "reliability_diagram",
    "RidgelinePlot": "ridgeline",
    "ROCCurvePlot": "roc_curve",
    "ScatterPlot": "scatter",
    "SoccerPlot": "soccer",
    "SpatialBiasScatterPlot": "spatial_bias_scatter",
    "SpatialContourPlot": "spatial_contour",
    "SpatialFacetGridPlot": "facet_grid_spatial",
    "SpatialPlot": "spatial",
    "SpreadSkillPlot": "ensemble",
    "TaylorDiagramPlot": "taylor_diagram",
    "TimeSeriesPlot": "timeseries",
    "TrajectoryPlot": "trajectory",
    "UpperAir": "upper_air",
    "WindBarbsPlot": "wind_barbs",
    "WindQuiverPlot": "wind_quiver",
    "Windrose": "windrose",
    "categorical_plot": "categorical",
}

FIXTURES = {
    "RadarPlot": "2 models × 4 metrics",
    "ScorecardPlot": "4 regions × 12 lead times",
    "SpatialImshowPlot": "40 × 50 cells",
    "SpatialTrack": "200 track points",
    "StickPlot": "36 wind vectors",
    "TaylorDiagram": "3 samples and 5 RMS contours",
    "TemporalScorecardPlot": "4 regions × 12 dates",
    "TimeSeriesErrorBarPlot": "240 observations in 10 daily bins",
    "TimeSeriesStatsPlot": "240 paired observations in 10 daily bins",
    "VerticalBoxPlot": "100 values in 5 altitude bins",
    "VerticalSlice": "50 × 50 grid",
}

MAP_HELPERS = {
    "facet_time_map": "4 map panels with 40 × 50 cells each",
    "plot_lines_map": "200 track points",
    "plot_points_map": "200 track points",
    "plot_quick_contourf": "40 × 50 cells",
    "plot_quick_imshow": "40 × 50 cells",
    "plot_quick_map": "40 × 50 cells",
}

EXCLUSIONS = {
    "BasePlot": "plot infrastructure with no plot method",
    "SpScatterBiasPlot": "listed in __all__ but no such implementation or alias",
    "categorical_timeseries": "placeholder that raises NotImplementedError",
}


def check_inventory():
    """Find future uncovered concrete plot classes before running the suite."""
    classes = {
        node.name
        for path in (ROOT / "src/monet_plots/plots").glob("*.py")
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name not in EXCLUSIONS
    }
    expected = classes | {"TaylorDiagram", "categorical_plot"}
    registered = GALLERY.keys() | FIXTURES.keys()
    if expected != registered:
        raise RuntimeError(
            f"Update plot inventory. Missing: {sorted(expected - registered)}. "
            f"Unrecognized: {sorted(registered - expected)}."
        )


def draw_fixture(name):
    """Supply small deterministic data where no single-plot gallery case exists."""
    from types import SimpleNamespace

    import numpy as np
    import pandas as pd
    import xarray as xr

    from monet_plots import cartopy_utils
    from monet_plots.plots.profile import StickPlot, VerticalBoxPlot, VerticalSlice
    from monet_plots.plots.radar import RadarPlot
    from monet_plots.plots.scorecard import ScorecardPlot
    from monet_plots.plots.spatial import SpatialTrack
    from monet_plots.plots.spatial_imshow import SpatialImshowPlot
    from monet_plots.plots.temporal_scorecard import TemporalScorecardPlot
    from monet_plots.plots.timeseries import TimeSeriesErrorBarPlot, TimeSeriesStatsPlot
    from monet_plots.taylordiagram import TaylorDiagram

    rng = np.random.default_rng(42)
    map_features = {"coastlines": {"resolution": "50m"}}
    if name == "RadarPlot":
        metrics = xr.Dataset(
            {
                metric: ("model", values)
                for metric, values in {
                    "Correlation": [0.86, 0.72],
                    "Accuracy": [0.69, 0.75],
                    "Reliability": [0.89, 0.63],
                    "Discrimination": [0.79, 0.82],
                }.items()
            },
            coords={"model": ["One", "Two"]},
        )
        RadarPlot(metrics_data=metrics, style="journal").plot()
    elif name in {"ScorecardPlot", "TemporalScorecardPlot"}:
        columns = (
            pd.date_range("2025-01-01", periods=12, freq="D")
            if name == "TemporalScorecardPlot"
            else np.arange(6, 73, 6)
        )
        data = pd.MultiIndex.from_product(
            [["North", "South", "East", "West"], columns],
            names=["region", "column"],
        ).to_frame(index=False)
        data["score"] = rng.normal(size=len(data))
        if name == "TemporalScorecardPlot":
            TemporalScorecardPlot().plot(data, "column", "region", "score")
        else:
            ScorecardPlot(style="journal").plot(data, "column", "region", "score")
    elif name in {
        "SpatialImshowPlot",
        "plot_quick_contourf",
        "plot_quick_imshow",
        "plot_quick_map",
        "facet_time_map",
    }:
        lat = np.linspace(27, 49, 40)
        lon = np.linspace(-125, -65, 50)
        xx, yy = np.meshgrid(lon, lat)
        data = xr.DataArray(
            np.cos(xx / 5) + np.sin(yy / 5),
            coords={"lat": lat, "lon": lon},
            dims=("lat", "lon"),
        )
        if name == "SpatialImshowPlot":
            grid = SimpleNamespace(
                variables={"LAT": yy[None, None], "LON": xx[None, None]}
            )
            SpatialImshowPlot(data.values, grid, coastlines=False).plot()
        elif name == "facet_time_map":
            facets = xr.concat(
                [data + t for t in range(4)],
                dim=pd.Index(
                    pd.date_range("2025-01-01", periods=4, freq="D"), name="time"
                ),
            )
            cartopy_utils.facet_time_map(facets, map_kws=map_features)
        else:
            getattr(cartopy_utils, name)(data, map_kws=map_features)
    elif name in {"SpatialTrack", "plot_lines_map", "plot_points_map"}:
        count = 200
        lon = np.linspace(-116, -95, count)
        lat = 36 + np.sin(lon / 4)
        if name == "SpatialTrack":
            data = xr.DataArray(
                np.cos(lon / 7),
                dims="time",
                coords={"lat": ("time", lat), "lon": ("time", lon)},
            )
            SpatialTrack(data, coastlines=False).plot()
        else:
            data = pd.DataFrame({"longitude": lon, "latitude": lat})
            getattr(cartopy_utils, name)(data, map_kws=map_features)
    elif name == "StickPlot":
        StickPlot(rng.normal(size=36), rng.normal(size=36), np.arange(36)).plot()
    elif name == "TaylorDiagram":
        dia = TaylorDiagram(1.0, label="Reference")
        for standard_deviation, correlation in [(0.75, 0.6), (1.1, 0.9), (1.3, 0.95)]:
            dia.add_sample(standard_deviation, correlation)
        dia.add_contours(levels=5)
    elif name in {"TimeSeriesErrorBarPlot", "TimeSeriesStatsPlot"}:
        observations = rng.normal(25, 3, size=240)
        data = pd.DataFrame(
            {
                "time": pd.date_range("2025-01-01", periods=240, freq="h"),
                "obs": observations,
                "model": observations + rng.normal(size=240),
            }
        )
        if name == "TimeSeriesStatsPlot":
            TimeSeriesStatsPlot(data, var1="obs", var2="model", x="time").plot(
                stat="bias", freq="D"
            )
        else:
            TimeSeriesErrorBarPlot(data, x="time", y="obs", freq="D").plot()
    elif name == "VerticalBoxPlot":
        VerticalBoxPlot(
            rng.normal(size=100), np.linspace(0, 10, 100), range(0, 11, 2)
        ).plot()
    elif name == "VerticalSlice":
        xx, yy = np.meshgrid(np.linspace(0, 10, 50), np.linspace(0, 10, 50))
        VerticalSlice(x=xx, y=yy, z=np.sin(xx) * np.cos(yy)).plot()
    else:
        raise KeyError(f"No benchmark fixture for {name}")


def benchmark(case, cartopy_data):
    import cartopy
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.figure import Figure

    importlib.import_module("monet_plots")
    matplotlib.use("Agg", force=True)
    if cartopy_data is not None:
        cartopy.config["data_dir"] = cartopy_data
    np.random.seed(42)
    rendered = set()
    draw_seconds = 0.0

    def draw(fig, *_args, **_kwargs):
        nonlocal draw_seconds
        if fig not in rendered:
            start = perf_counter()
            fig.canvas.draw()
            draw_seconds += perf_counter() - start
            rendered.add(fig)

    def draw_all(*_args, **_kwargs):
        for number in plt.get_fignums():
            draw(plt.figure(number))

    start = perf_counter()
    with patch.object(Figure, "savefig", draw), patch.object(plt, "show", draw_all):
        if case in GALLERY:
            runpy.run_path(
                str(EXAMPLES / f"plot_{GALLERY[case]}.py"), run_name="__main__"
            )
        else:
            draw_fixture(case)
        draw_all()
    elapsed = perf_counter() - start
    if not rendered:
        raise RuntimeError("No figure was rendered")
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_divisor = 1024**2 if sys.platform == "darwin" else 1024
    return {
        "first_render_ms": elapsed * 1000,
        "canvas_draw_ms": draw_seconds * 1000,
        "peak_rss_mib": peak_rss / rss_divisor,
        "figures_rendered": len(rendered),
        "matplotlib": matplotlib.__version__,
    }


def run_worker(case, cartopy_data, timeout, source_root):
    command = [
        sys.executable,
        __file__,
        "--worker",
        case,
        "--source-root",
        str(source_root),
    ]
    if cartopy_data is not None:
        command.extend(["--cartopy-data", str(cartopy_data)])
    env = dict(
        os.environ,
        MPLBACKEND="Agg",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONHASHSEED="0",
    )
    try:
        result = subprocess.run(
            command, env=env, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "error": f"exceeded {timeout:g}s worker timeout"}
    if result.returncode != 0:
        detail = [line.strip() for line in result.stderr.splitlines() if line.strip()]
        return {
            "status": "failed",
            "error": detail[-1] if detail else f"exit {result.returncode}",
        }
    return {"status": "ok", **json.loads(result.stdout.splitlines()[-1])}


def measure_child(case, cartopy_data, output):
    with output:
        try:
            result = {"status": "ok", **benchmark(case, cartopy_data)}
        except Exception as exc:
            result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        output.send(result)


def run_preloaded(case, cartopy_data, timeout, context):
    reader, writer = context.Pipe(duplex=False)
    child = context.Process(target=measure_child, args=(case, cartopy_data, writer))
    with reader, writer:
        child.start()
        writer.close()
        if not reader.poll(timeout):
            child.terminate()
            child.join()
            return {
                "status": "timeout",
                "error": f"exceeded {timeout:g}s worker timeout",
            }
        try:
            result = reader.recv()
        except EOFError:
            result = {"status": "failed", "error": "child exited without a result"}
        child.join()
        return result


def source_for(case):
    if case in GALLERY:
        return f"examples/plot_{GALLERY[case]}.py"
    return FIXTURES[case] if case in FIXTURES else MAP_HELPERS[case]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    names = sorted(GALLERY.keys() | FIXTURES.keys() | MAP_HELPERS.keys())
    parser.add_argument("--case", action="append", choices=names)
    parser.add_argument("--worker", choices=names, help=argparse.SUPPRESS)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=ROOT / "src",
        help="Directory containing monet_plots, defaults to the checkout src directory",
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Fork fresh workers after loading shared imports on platforms with fork",
    )
    parser.add_argument(
        "--cartopy-data", type=Path, help="Directory of cached Cartopy data"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("static-plot-benchmark.csv")
    )
    parser.add_argument(
        "--list", action="store_true", help="List workloads and exclusions"
    )
    args = parser.parse_args()
    if args.repeats < 1 or args.timeout <= 0:
        parser.error("--repeats and --timeout must be positive")
    if args.preload and "fork" not in multiprocessing.get_all_start_methods():
        parser.error("--preload requires multiprocessing fork support")
    return args, names


def main():
    args, names = parse_args()
    source_root = args.source_root.resolve()
    if not (source_root / "monet_plots" / "__init__.py").is_file():
        raise SystemExit(f"MONET Plots source not found in {source_root}")
    sys.path.insert(0, str(source_root))
    if args.worker:
        print(json.dumps(benchmark(args.worker, args.cartopy_data)))
        return 0
    check_inventory()
    if args.list:
        for case in args.case or names:
            print(f"{case}: {source_for(case)}")
        print("Excluded:")
        for name, reason in EXCLUSIONS.items():
            print(f"  {name}: {reason}")
        return 0

    fields = [
        "case",
        "scope",
        "source",
        "repetition",
        "status",
        "first_render_ms",
        "canvas_draw_ms",
        "peak_rss_mib",
        "figures_rendered",
        "matplotlib",
        "error",
        "source_root",
        "mode",
    ]
    any_failures = False
    context = None
    if args.preload:
        os.environ.update(
            MPLBACKEND="Agg",
            OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
        )
        importlib.import_module("monet_plots")
        context = multiprocessing.get_context("fork")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for case in args.case or names:
            elapsed = []
            peaks = []
            for repetition in range(1, args.repeats + 1):
                if context is not None:
                    result = run_preloaded(
                        case, args.cartopy_data, args.timeout, context
                    )
                else:
                    result = run_worker(
                        case, args.cartopy_data, args.timeout, source_root
                    )
                writer.writerow(
                    {
                        "case": case,
                        "source": source_for(case),
                        "scope": "map helper" if case in MAP_HELPERS else "plot type",
                        "repetition": repetition,
                        "source_root": source_root,
                        "mode": "preloaded-fork" if context is not None else "exec",
                        **result,
                    }
                )
                output.flush()
                if result["status"] == "ok":
                    elapsed.append(result["first_render_ms"])
                    peaks.append(result["peak_rss_mib"])
                else:
                    any_failures = True
                    print(
                        f"{case} [{repetition}]: {result['status']}: {result['error']}",
                        flush=True,
                    )
            if elapsed:
                print(
                    f"{case}: median {statistics.median(elapsed):.1f} ms, "
                    f"max peak RSS {max(peaks):.1f} MiB ({len(elapsed)}/{args.repeats} passed)",
                    flush=True,
                )
    return int(any_failures)


if __name__ == "__main__":
    sys.exit(main())
