"""Check benchmark isolation and measurements with a tiny, offline case."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_isolated_benchmark_worker(tmp_path):
    pytest.importorskip("psutil")
    script = Path(__file__).resolve().parents[1] / "benchmarks/bench_spatial.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--profile",
            "smoke",
            "--case",
            "comparison_explicit",
            "--repeat",
            "1",
            "--output",
            str(tmp_path / "results.json"),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
        env={**os.environ, "MPLBACKEND": "Agg"},
    )
    assert "comparison_explicit" in result.stdout
    report = json.loads((tmp_path / "results.json").read_text())
    run = report["results"][0]["runs"][0]
    assert run["panels"] == 3 and run["displayed_cells"] == 3 * 32**2
    assert run["render_s"] > 0 and run["png_s"] > 0
    assert run["peak_rss_mib"] >= run["rss_baseline_mib"] > 0
    assert run["dask_tasks"]["construct"] > 0
    assert run["dask_tasks"]["draw"] == run["dask_tasks"]["png"] == 0
    assert run["png_bytes"] > 1000
    assert (tmp_path / "results.md").exists()
