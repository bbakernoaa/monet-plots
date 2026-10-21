"""Reproducible, isolated-process spatial rendering benchmarks.

Run: python benchmarks/bench_spatial.py --profile standard --repeat 2 --output artifacts/benchmarks.json
No scientific datasets, map features, network access, or GPU are needed.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from importlib.metadata import version
from pathlib import Path


class MemorySampler:
    """Sample process RSS, including native NumPy/Matplotlib memory, every 10 ms."""

    def __enter__(self):
        import psutil

        self.process = psutil.Process()
        self.baseline = self.peak = self.process.memory_info().rss
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def _sample(self):
        while not self.done.wait(0.01):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __exit__(self, *args):
        self.peak = max(self.peak, self.process.memory_info().rss)
        self.done.set()
        self.thread.join()


def cases(profile):
    size = 32 if profile == "smoke" else 1024
    base = {
        "size": size,
        "facets": 1,
        "backend": "dask",
        "renderer": "pcolormesh",
        "scale": "auto",
        "suite": "maps",
        "chunks": min(256, size),
    }
    result = []
    variants = [
        ("mesh_numpy", {"backend": "numpy"}),
        ("mesh_dask", {}),
        ("facets_auto", {"facets": 4}),
        ("facets_explicit", {"facets": 4, "scale": "explicit"}),
        ("images_explicit", {"facets": 4, "renderer": "imshow", "scale": "explicit"}),
        ("comparison_auto", {"suite": "comparison"}),
        ("comparison_explicit", {"suite": "comparison", "scale": "explicit"}),
        ("comparison_rows", {"suite": "comparison", "facets": 4, "scale": "explicit"}),
    ]
    if profile == "standard":
        variants.extend(
            [
                (
                    "comparison_panel_auto",
                    {"suite": "comparison", "materialize": "panel"},
                ),
                (
                    "comparison_panel_explicit",
                    {
                        "suite": "comparison",
                        "materialize": "panel",
                        "scale": "explicit",
                    },
                ),
            ]
        )
        variants.append(("mesh_2048", {"size": 2048, "scale": "explicit"}))
    for name, overrides in variants:
        result.append({**base, "name": name, **overrides})
    return result


def make_fields(case):
    import numpy as np
    import xarray as xr

    size = case["size"]
    lon = np.linspace(-125, -65, size, dtype=np.float32)
    lat = np.linspace(25, 50, size, dtype=np.float32)
    if case["backend"] == "dask":
        import dask.array as da

        xx = da.from_array(lon, chunks=case["chunks"])[None, :]
        yy = da.from_array(lat, chunks=case["chunks"])[:, None]
        array = da
    else:
        xx, yy, array = lon[None, :], lat[:, None], np
    values = (30 + 15 * array.cos((xx + 95) / 18) * array.sin((yy - 20) / 15)).astype(
        "float32"
    )
    obs = xr.DataArray(
        values,
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
        name="ozone",
        attrs={"units": "ppb", "long_name": "Ozone"},
    )
    models = [obs + np.float32(2 + i) for i in range(case["facets"])]
    model = (
        models[0]
        if len(models) == 1
        else xr.concat(models, dim=xr.IndexVariable("model", range(len(models))))
    )
    return obs, model


def run_case(case):
    import dask
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import monet_plots as mp
    from dask.callbacks import Callback

    # Prime plotting imports and font caches outside the measured region.
    warm = {"size": 8, "facets": 1, "backend": "numpy", "chunks": 8}
    _, data = make_fields(warm)
    plot = mp.spatial(data, coastlines=False, style=None, colorbar=False)
    plot.fig.canvas.draw()
    plot.close()
    tasks = {"input": 0, "construct": 0, "draw": 0, "png": 0}
    stage = "input"

    def completed(*args):
        tasks[stage] += 1

    with (
        dask.config.set(scheduler="synchronous"),
        plt.rc_context({"figure.dpi": 80, "savefig.dpi": 80}),
        MemorySampler() as memory,
        Callback(posttask=completed),
    ):
        started = time.perf_counter()
        obs, model = make_fields(case)
        setup_s = time.perf_counter() - started
        stage = "construct"
        started = time.perf_counter()
        bounds = {"vmin": 0, "vmax": 80} if case["scale"] == "explicit" else {}
        common = {
            "coastlines": False,
            "style": None,
            "size": 2.4,
            "aspect": 1.2,
            **bounds,
        }
        if case["suite"] == "comparison":
            if case["scale"] == "explicit":
                common["bias_limit"] = 10
            common["materialize"] = case.get("materialize", "row")
            result = mp.compare_spatial(
                obs,
                model,
                row="model" if case["facets"] > 1 else None,
                kind=case["renderer"],
                **common,
            )
            panels = case["facets"] * 3
        else:
            result = mp.spatial(
                model,
                col="model" if case["facets"] > 1 else None,
                col_wrap=2 if case["facets"] > 1 else None,
                kind=case["renderer"],
                **common,
            )
            panels = case["facets"]
        construct_s = time.perf_counter() - started
        stage = "draw"
        started = time.perf_counter()
        result.fig.canvas.draw()
        draw_s = time.perf_counter() - started
        stage = "png"
        output = io.BytesIO()
        started = time.perf_counter()
        result.save(output, format="png", dpi=80)
        png_s = time.perf_counter() - started
        png_bytes = output.tell()
        resident_before_close = memory.process.memory_info().rss
        result.close()
    return {
        "input_s": setup_s,
        "construct_s": construct_s,
        "draw_s": draw_s,
        "png_s": png_s,
        "render_s": construct_s + draw_s,
        "total_s": setup_s + construct_s + draw_s + png_s,
        "rss_baseline_mib": memory.baseline / 2**20,
        "peak_rss_mib": memory.peak / 2**20,
        "peak_rss_increase_mib": (memory.peak - memory.baseline) / 2**20,
        "rss_before_close_mib": resident_before_close / 2**20,
        "dask_tasks": tasks,
        "panels": panels,
        "displayed_cells": panels * case["size"] ** 2,
        "png_bytes": png_bytes,
    }


def environment():
    import psutil

    packages = [
        "monet-plots",
        "numpy",
        "xarray",
        "dask",
        "matplotlib",
        "cartopy",
        "psutil",
    ]
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted((root / "src/monet_plots").rglob("*.py")) + [
        Path(__file__).resolve()
    ]:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return {
        "source_sha256": digest.hexdigest(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "available_memory_mib": psutil.virtual_memory().available / 2**20,
        "packages": {name: version(name) for name in packages},
        "scheduler": "synchronous",
        "numeric_threads": 1,
        "rss_sample_ms": 10,
        "dtype": "float32",
        "dpi": 80,
        "warmup_grid": "8x8",
    }


def summarize(case, runs):
    metrics = [
        "input_s",
        "construct_s",
        "draw_s",
        "png_s",
        "render_s",
        "total_s",
        "peak_rss_mib",
        "peak_rss_increase_mib",
    ]
    return {
        "case": case,
        "median": {key: statistics.median(r[key] for r in runs) for key in metrics},
        "max_peak_rss_mib": max(r["peak_rss_mib"] for r in runs),
        "runs": runs,
    }


def report_markdown(report):
    text = [
        "# Spatial rendering benchmark",
        "",
        f"Python {report['environment']['python']}; {report['environment']['platform']}; {report['repeat']} isolated repetitions per case.",
        "",
        "Times are medians. Render = scale selection/artist construction + an actual canvas draw; PNG is a second draw/export.",
        "RSS includes native allocations. Peak increase is relative to warmed imports before creating inputs; sampled peaks may miss allocations shorter than 10 ms.",
        "",
        "| Case | Grid | Panels | Render (s) | PNG (s) | Peak RSS (MiB) | Peak increase (MiB) | Dask tasks: construct / draw / PNG |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in report["results"]:
        case, med, run = result["case"], result["median"], result["runs"][0]
        tasks = run["dask_tasks"]
        text.append(
            f"| {case['name']} | {case['size']}² | {run['panels']} | {med['render_s']:.3f} | {med['png_s']:.3f} | {med['peak_rss_mib']:.1f} | {med['peak_rss_increase_mib']:.1f} | {tasks['construct']} / {tasks['draw']} / {tasks['png']} |"
        )
    text += [
        "",
        "Synthetic fields, resident/lazy input generation, geographic PlateCarree, Agg backend, no map features or file/network I/O. Input generation is reported separately in JSON. These are machine-specific measurements, not performance guarantees.",
        "",
        "Automatic scales perform global reductions; explicit limits avoid them. All displayed cells still enter Matplotlib. Rasterizing vector output does not reduce the number of in-memory cells. Dask does not make a Matplotlib figure out-of-core.",
        "",
    ]
    return "\n".join(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["smoke", "standard"], default="smoke")
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument(
        "--case", action="append", help="Run only named cases; may be repeated"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/benchmarks.json")
    )
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(run_case(json.loads(args.worker))))
        return
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    selected = [
        c for c in cases(args.profile) if not args.case or c["name"] in args.case
    ]
    if not selected or args.case and set(args.case) - {c["name"] for c in selected}:
        parser.error("Unknown case name; inspect cases() or use a default profile")
    report = {
        "environment": environment(),
        "profile": args.profile,
        "repeat": args.repeat,
        "results": [],
    }
    env = {
        **os.environ,
        "MPLBACKEND": "Agg",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    for case in selected:
        runs = []
        for repeat in range(args.repeat):
            print(f"{case['name']} ({repeat + 1}/{args.repeat})", flush=True)
            worker = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    json.dumps(case),
                ],
                capture_output=True,
                text=True,
                env=env,
                timeout=300,
            )
            if worker.returncode:
                raise RuntimeError(f"{case['name']} failed:\n{worker.stderr}")
            runs.append(json.loads(worker.stdout))
        report["results"].append(summarize(case, runs))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        args.output.with_suffix(".md").write_text(report_markdown(report))
    print(f"Results: {args.output} and {args.output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
