"""Headless export and repeatability checks for the scientific plot gallery."""

import runpy
from pathlib import Path

import numpy as np
from PIL import Image


def test_scientific_gallery_exports_reproducibly(tmp_path):
    script = Path(__file__).resolve().parents[2] / "examples/plot_scientific_quality.py"
    render = runpy.run_path(str(script))["render"]
    first, second = tmp_path / "first", tmp_path / "second"
    render(first, "operations")
    render(second, "operations")
    for name in ("timeseries", "station_bias", "reliability", "taylor"):
        with Image.open(first / f"{name}.png") as image:
            assert image.size == (1500, 900)
            np.testing.assert_array_equal(
                np.asarray(image), np.asarray(Image.open(second / f"{name}.png"))
            )
        assert (first / f"{name}.csv").stat().st_size
    assert (first / "taylor_statistics.csv").stat().st_size
