import dask.array as da
import numpy as np
import pytest
import xarray as xr

from monet_plots.plots.conditional_bias import ConditionalBiasPlot
from monet_plots.verification_metrics import compute_binned_bias


def test_compute_binned_bias_lazy():
    """Verify compute_binned_bias works with lazy dask inputs."""
    obs_data = np.linspace(0, 10, 100)
    mod_data = obs_data + 1.0

    obs = xr.DataArray(da.from_array(obs_data, chunks=25), dims=["x"], name="obs")
    mod = xr.DataArray(da.from_array(mod_data, chunks=25), dims=["x"], name="mod")

    stats = compute_binned_bias(obs, mod, n_bins=5)

    # Check if lazy
    assert stats.bias_mean.chunks is not None

    # Compute and verify
    res = stats.compute()
    assert len(res.bin_center) == 5
    np.testing.assert_allclose(res.bias_mean, 1.0)
    assert "Calculated binned bias" in res.attrs["history"]


def test_conditional_bias_eager_lazy_parity():
    """Verify results are identical for Eager and Lazy backends."""
    obs_data = np.random.rand(100)
    mod_data = obs_data + np.random.normal(0, 0.1, 100)

    # Eager
    obs_e = xr.DataArray(obs_data, dims=["x"])
    mod_e = xr.DataArray(mod_data, dims=["x"])
    stats_e = compute_binned_bias(obs_e, mod_e, n_bins=5).compute()

    # Lazy
    obs_l = xr.DataArray(da.from_array(obs_data, chunks=50), dims=["x"])
    mod_l = xr.DataArray(da.from_array(mod_data, chunks=50), dims=["x"])
    stats_l = compute_binned_bias(obs_l, mod_l, n_bins=5).compute()

    xr.testing.assert_allclose(stats_e, stats_l)


def test_conditional_bias_plot_lazy():
    """Verify ConditionalBiasPlot handles lazy xarray Dataset."""
    obs_data = np.random.rand(100)
    mod_data = obs_data + 0.5

    ds = xr.Dataset(
        {
            "observation": (["time"], da.from_array(obs_data, chunks=50)),
            "forecast": (["time"], da.from_array(mod_data, chunks=50)),
        }
    )

    plot_obj = ConditionalBiasPlot(data=ds)
    ax = plot_obj.plot(obs_col="observation", fcst_col="forecast", n_bins=5)

    assert ax is not None
    # Check if we have error bars (usually represented as lines in Matplotlib)
    assert len(ax.get_lines()) >= 1


def test_conditional_bias_hvplot():
    """Verify hvplot method returns a Holoviews object."""
    pytest.importorskip("holoviews")
    pytest.importorskip("hvplot")

    obs_data = np.random.rand(100)
    mod_data = obs_data + 0.5
    ds = xr.Dataset({"obs": (["x"], obs_data), "mod": (["x"], mod_data)})

    plot_obj = ConditionalBiasPlot(data=ds)
    hv_plot = plot_obj.hvplot(obs_col="obs", fcst_col="mod", n_bins=5)

    import holoviews as hv

    assert isinstance(hv_plot, hv.core.overlay.Overlay)


def test_conditional_bias_grouping():
    """Verify ConditionalBiasPlot handles grouping with label_col."""
    obs_data = np.random.rand(100)
    mod_data = obs_data + 0.5
    labels = ["Model A"] * 50 + ["Model B"] * 50

    ds = xr.Dataset(
        {
            "obs": (["x"], obs_data),
            "mod": (["x"], mod_data),
            "label": (["x"], labels),
        }
    )

    plot_obj = ConditionalBiasPlot(data=ds)
    ax = plot_obj.plot(obs_col="obs", fcst_col="mod", label_col="label", n_bins=5)

    assert ax is not None
    # Check the legend
    legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "Model A" in legend_texts
    assert "Model B" in legend_texts


def test_compute_binned_bias_bin_range():
    """Verify compute_binned_bias works with explicit bin_range (fully lazy)."""
    obs_data = np.linspace(0, 10, 100)
    mod_data = obs_data + 1.0

    obs = xr.DataArray(da.from_array(obs_data, chunks=25), dims=["x"], name="obs")
    mod = xr.DataArray(da.from_array(mod_data, chunks=25), dims=["x"], name="mod")

    stats = compute_binned_bias(obs, mod, n_bins=5, bin_range=(0, 10))

    res = stats.compute()
    assert len(res.bin_center) == 5
    np.testing.assert_allclose(res.bias_mean, 1.0)
