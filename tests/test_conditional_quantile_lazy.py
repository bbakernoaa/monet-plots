# tests/test_conditional_quantile_lazy.py
import dask.array as da
import numpy as np
import pytest
import xarray as xr
from monet_plots.plots.conditional_quantile import ConditionalQuantilePlot
from monet_plots.verification_metrics import compute_binned_quantiles


def test_conditional_quantile_lazy_consistency():
    """Verify that eager and lazy execution yield the same results."""
    # Create sample data
    np.random.seed(42)
    obs_vals = np.linspace(0, 100, 100)
    mod_vals = obs_vals + np.random.normal(0, 5, 100)

    # Eager Xarray
    ds_eager = xr.Dataset({"obs": (["index"], obs_vals), "mod": (["index"], mod_vals)})

    # Lazy Xarray (Dask)
    ds_lazy = xr.Dataset(
        {
            "obs": (["index"], da.from_array(obs_vals, chunks=20)),
            "mod": (["index"], da.from_array(mod_vals, chunks=20)),
        }
    )

    # Verify the underlying metric function first
    res_eager = compute_binned_quantiles(ds_eager.obs, ds_eager.mod, n_bins=5)
    res_lazy = compute_binned_quantiles(ds_lazy.obs, ds_lazy.mod, n_bins=5)

    xr.testing.assert_allclose(res_eager, res_lazy.compute())
    assert "history" in res_eager.attrs
    assert "(monet-plots)" in res_eager.attrs["history"]

    # Plot with eager data
    plot_eager = ConditionalQuantilePlot(ds_eager, obs_col="obs", mod_col="mod", bins=5)
    ax_eager = plot_eager.plot()

    # Plot with lazy data
    plot_lazy = ConditionalQuantilePlot(ds_lazy, obs_col="obs", mod_col="mod", bins=5)
    ax_lazy = plot_lazy.plot()

    assert ax_eager is not None
    assert ax_lazy is not None


def test_conditional_quantile_provenance():
    """Verify that the history attribute is updated."""
    obs_vals = np.linspace(0, 100, 100)
    mod_vals = obs_vals + np.random.normal(0, 5, 100)
    ds = xr.Dataset({"obs": (["index"], obs_vals), "mod": (["index"], mod_vals)})

    stats = compute_binned_quantiles(ds.obs, ds.mod)
    assert "history" in stats.attrs
    assert "Calculated binned quantiles" in stats.attrs["history"]


def test_conditional_quantile_hvplot():
    """Verify Track B visualization."""
    pytest.importorskip("hvplot")
    obs_vals = np.linspace(0, 100, 100)
    mod_vals = obs_vals + np.random.normal(0, 5, 100)
    ds = xr.Dataset({"obs": (["index"], obs_vals), "mod": (["index"], mod_vals)})

    plot = ConditionalQuantilePlot(ds, obs_col="obs", mod_col="mod")
    hplot = plot.hvplot()
    assert hplot is not None
