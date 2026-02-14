from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

try:
    import dask.array as da
except ImportError:
    da = None
import matplotlib.pyplot as plt

from monet_plots.plots.performance_diagram import PerformanceDiagramPlot


@pytest.mark.skipif(da is None, reason="dask not installed")
def test_performance_diagram_lazy_parity():
    """Verify PerformanceDiagramPlot handles lazy data and matches eager results."""
    # Create test data (hits, misses, fa, cn)
    # Using a larger dataset to make it interesting
    shape = (10, 10, 10)
    obs = (np.random.rand(*shape) > 0.5).astype(float)
    mod = (np.random.rand(*shape) > 0.5).astype(float)

    ds_eager = xr.Dataset(
        {
            "obs": (["x", "y", "z"], obs),
            "mod": (["x", "y", "z"], mod),
        }
    )
    ds_lazy = ds_eager.chunk({"x": 5})

    # Track A: Eager (using threshold)
    plot_eager = PerformanceDiagramPlot()
    ax_eager = plot_eager.plot(
        ds_eager, obs_col="obs", mod_col="mod", threshold=0.5, dim=["x", "y", "z"]
    )
    assert ax_eager is not None

    # Extract plotted points
    line_eager = ax_eager.get_lines()[-1]
    sr_eager, pod_eager = line_eager.get_data()
    plt.close(ax_eager.figure)

    # Track B: Lazy
    plot_lazy = PerformanceDiagramPlot()
    ax_lazy = plot_lazy.plot(
        ds_lazy, obs_col="obs", mod_col="mod", threshold=0.5, dim=["x", "y", "z"]
    )
    assert ax_lazy is not None

    line_lazy = ax_lazy.get_lines()[-1]
    sr_lazy, pod_lazy = line_lazy.get_data()
    plt.close(ax_lazy.figure)

    # Parity check
    np.testing.assert_allclose(sr_eager, sr_lazy)
    np.testing.assert_allclose(pod_eager, pod_lazy)

    # History check
    assert "Plotted with PerformanceDiagramPlot" in ds_lazy.attrs["history"]


@pytest.mark.skipif(da is None, reason="dask not installed")
def test_performance_diagram_precalculated():
    """Verify PerformanceDiagramPlot works with pre-calculated metrics in Xarray."""
    ds = xr.Dataset(
        {
            "success_ratio": (["label"], [0.8, 0.6]),
            "pod": (["label"], [0.7, 0.9]),
        },
        coords={"label": ["Model A", "Model B"]},
    )
    ds_lazy = ds.chunk({"label": 1})

    plot = PerformanceDiagramPlot()
    ax = plot.plot(ds_lazy, label_col="label")
    assert ax is not None

    # Should have 2 points plotted (as individual lines or markers)
    # Since label_col is used, it plots each group
    assert len(ax.get_lines()) >= 2
    plt.close(ax.figure)
