"""Check eager and Dask-backed inputs at public static plot entry points."""

import argparse
from contextlib import ExitStack
from functools import wraps
from importlib import import_module
from inspect import Parameter, signature
import json
import os
from pathlib import Path
from pkgutil import iter_modules
import sys
from unittest.mock import patch

if __package__:
    from . import benchmark_plot_types as suite
else:
    import benchmark_plot_types as suite

# The base map builder's inputs are geometry settings. The low-level Taylor
# diagram accepts scalar sample statistics. The corresponding data APIs run below.
NO_DATA_INPUT = {"SpatialPlot", "TaylorDiagram"}
INPUT_NAMES = {
    "data",
    "df",
    "da",
    "modelvar",
    "metrics_data",
    "spread",
    "skill",
    "x",
    "y",
    "z",
    "u",
    "v",
    "ws",
    "wd",
    "wdir",
    "lat",
    "lon",
    "hgt",
    "longitude",
    "latitude",
    "time",
    "ts_data",
    "var1_data",
}
CASES = sorted(suite.GALLERY.keys() | suite.FIXTURES.keys() | suite.MAP_HELPERS.keys())


def chunk_inputs(stack, name, converted):
    """Substitute the equivalent chunked object before the real API call."""
    import dask.array as da
    import dask.dataframe as dd
    import numpy as np
    import pandas as pd
    import xarray as xr

    def chunk(value):
        if isinstance(value, (pd.DataFrame, pd.Series)):
            result = dd.from_pandas(value, npartitions=3, sort=False)
        elif isinstance(value, (xr.DataArray, xr.Dataset)):
            result = value.chunk(
                {dim: max(1, length // 3) for dim, length in value.sizes.items()}
            )
        elif isinstance(value, np.ndarray) and value.ndim and value.dtype.kind != "O":
            result = da.from_array(
                value, chunks=tuple(max(1, n // 3) for n in value.shape)
            )
        else:
            return value
        converted.append(type(result).__name__)
        return result

    def install(module, method):
        original = getattr(module, method)
        inputs = signature(original)

        @wraps(original)
        def wrapper(*args, **kwargs):
            bound = inputs.bind(*args, **kwargs)
            for key, value in bound.arguments.items():
                if key in INPUT_NAMES:
                    bound.arguments[key] = chunk(value)
                elif inputs.parameters[key].kind == Parameter.VAR_KEYWORD:
                    bound.arguments[key] = {
                        item: chunk(val) if item in INPUT_NAMES else val
                        for item, val in value.items()
                    }
            return original(*bound.args, **bound.kwargs)

        stack.enter_context(patch.object(module, method, wrapper))

    if name in suite.MAP_HELPERS:
        install(import_module("monet_plots.cartopy_utils"), name)
    elif name == "categorical_plot":
        install(import_module("monet_plots.plots.categorical"), name)
    else:
        package = import_module("monet_plots.plots")
        for spec in iter_modules(package.__path__, prefix=package.__name__ + "."):
            module = import_module(spec.name)
            if hasattr(module, name):
                cls = getattr(module, name)
                install(cls, "__init__")
                if hasattr(cls, "plot"):
                    install(cls, "plot")
                return
        raise KeyError(name)


def compare(name):
    import dask
    import matplotlib.pyplot as plt
    import numpy as np
    from cartopy.mpl.geoaxes import GeoAxes
    from dask.callbacks import Callback

    if name in NO_DATA_INPUT:
        return {"case": name, "status": "no-data-input"}

    plt.close("all")
    images = [[], []]
    converted, runs = [], []
    try:
        for mode in (0, 1):
            with ExitStack() as stack:
                stack.enter_context(plt.rc_context())
                # Geometry files are orthogonal to data parity. This keeps the run offline.
                stack.enter_context(
                    patch.object(GeoAxes, "add_feature", return_value=None)
                )
                stack.enter_context(dask.config.set(scheduler="synchronous"))
                if mode:
                    chunk_inputs(stack, name, converted)
                    stack.enter_context(
                        Callback(start=lambda graph: runs.append(len(graph)))
                    )
                suite.benchmark(
                    name,
                    None,
                    observe=lambda fig: images[mode].append(
                        np.asarray(fig.canvas.buffer_rgba()).copy()
                    ),
                )
            plt.close("all")
        if not converted:
            raise AssertionError(
                "Fixture did not pass a chunkable input through the public API"
            )
        if not runs or not images[0]:
            raise AssertionError("No Dask work or rendered figure observed")
        if len(images[0]) != len(images[1]):
            raise AssertionError("Figure count changed")
        changed_fraction = 0.0
        for eager, chunked in zip(*images, strict=True):
            if eager.shape != chunked.shape:
                raise AssertionError(
                    f"Figure dimensions changed ({eager.shape} vs {chunked.shape})"
                )
            delta = np.abs(eager.astype(np.int16) - chunked)
            changed_fraction = max(
                changed_fraction, float(np.mean(np.max(delta, axis=2) > 3))
            )
        return {
            "case": name,
            "status": "ok" if changed_fraction <= 0.001 else "mismatch",
            "changed_pixel_fraction": changed_fraction,
            "inputs": sorted(set(converted)),
            "scheduler_runs": len(runs),
        }
    except Exception as exc:
        return {
            "case": name,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        plt.close("all")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=CASES)
    parser.add_argument(
        "--output", type=Path, default=Path("chunked-plot-parity.jsonl")
    )
    args = parser.parse_args()
    sys.path.insert(0, str(suite.ROOT / "src"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    suite.check_inventory()
    results = [compare(name) for name in args.case or CASES]
    args.output.write_text("".join(json.dumps(row) + "\n" for row in results))
    for result in results:
        print(json.dumps(result), flush=True)
    return int(any(row["status"] in {"mismatch", "error"} for row in results))


if __name__ == "__main__":
    raise SystemExit(main())
