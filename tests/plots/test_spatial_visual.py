"""A small offline image regression for facet titles, layout and shared legend."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.testing.compare import compare_images

import monet_plots as mp


def render_reference(path):
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 100,
            "savefig.dpi": 100,
            "savefig.bbox": None,
            "axes.grid": False,
        }
    ):
        data = xr.DataArray(
            np.arange(24).reshape(2, 3, 4),
            dims=("model", "lat", "lon"),
            coords={
                "model": ["A", "B"],
                "lat": [30, 35, 40],
                "lon": [-110, -100, -90, -80],
            },
            name="Ozone",
            attrs={"units": "ppb"},
        )
        result = mp.spatial(
            data,
            col="model",
            vmin=0,
            vmax=24,
            panel_labels=True,
            coastlines=False,
            style=None,
            size=2.4,
            aspect=1,
            cbar_kwargs={"orientation": "horizontal", "pad": 0.12},
        )
        result.save(path, dpi=100)
        result.close()


def test_spatial_comparison_image(tmp_path):
    actual = tmp_path / "comparison.png"
    render_reference(actual)
    baseline = Path(__file__).parent / "baseline_images" / "spatial_comparison.png"
    # Allow small font antialiasing differences across operating systems.
    assert compare_images(str(baseline), str(actual), tol=2.5) is None
