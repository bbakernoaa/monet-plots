import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from monet_plots.plots.conditional_bias import ConditionalBiasPlot
from monet_plots.verification_metrics import compute_binned_bias


@pytest.fixture
def clear_figures():
    """Clear all existing figures before and after a test."""
    plt.close("all")
    yield
    plt.close("all")


@pytest.fixture
def sample_data():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({"obs": np.random.rand(100), "fcst": np.random.rand(100)})


def test_conditional_bias_plot(clear_figures, sample_data):
    """Test ConditionalBiasPlot."""
    plot = ConditionalBiasPlot()
    plot.plot(data=sample_data, obs_col="obs", fcst_col="fcst")
    assert plot.ax is not None


def test_conditional_bias_plot_with_label_col(clear_figures):
    df = pd.DataFrame(
        {
            "obs": np.random.rand(100),
            "fcst": np.random.rand(100),
            "group": np.random.choice(["A", "B"], size=100),
        }
    )
    plot = ConditionalBiasPlot()
    plot.plot(data=df, obs_col="obs", fcst_col="fcst", label_col="group")
    # Should have a legend for groups
    assert plot.ax.get_legend() is not None


def test_conditional_bias_plot_n_bins(clear_figures, sample_data):
    plot = ConditionalBiasPlot()
    plot.plot(data=sample_data, obs_col="obs", fcst_col="fcst", n_bins=5)
    # Should have 5 or fewer points (bins with >=5 samples)
    lines = plot.ax.get_lines()
    assert any(len(line.get_xdata()) <= 5 for line in lines)


def test_conditional_bias_plot_empty_df(clear_figures):
    df = pd.DataFrame({"obs": [], "fcst": []})
    plot = ConditionalBiasPlot()
    with pytest.raises(ValueError):
        plot.plot(data=df, obs_col="obs", fcst_col="fcst")


def test_conditional_bias_plot_missing_column(clear_figures, sample_data):
    df = sample_data.drop(columns=["obs"])
    plot = ConditionalBiasPlot()
    with pytest.raises(ValueError):
        plot.plot(data=df, obs_col="obs", fcst_col="fcst")


def test_conditional_bias_plot_few_samples_per_bin(clear_figures):
    # Only 1 sample per bin, so no points should be plotted
    df = pd.DataFrame({"obs": np.arange(10), "fcst": np.arange(10) + 1})
    plot = ConditionalBiasPlot()
    plot.plot(data=df, obs_col="obs", fcst_col="fcst", n_bins=10)
    # No error, but no data points (lines) should be present except axhline
    lines = plot.ax.get_lines()
    # Only the zero-bias line should be present
    assert len(lines) == 1


def test_conditional_bias_zero_bias_line(clear_figures, sample_data):
    plot = ConditionalBiasPlot()
    plot.plot(data=sample_data, obs_col="obs", fcst_col="fcst")
    # Check for a horizontal line at y=0
    found = any(
        getattr(line, "get_ydata", lambda: [])()[0] == 0
        and all(y == 0 for y in line.get_ydata())
        for line in plot.ax.get_lines()
    )
    assert found


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


def test_conditional_bias_plot_dataarray(clear_figures):
    """Verify ConditionalBiasPlot handles DataArray input with explicit obs."""
    obs_data = np.random.rand(100)
    mod_data = obs_data + 0.5
    obs = xr.DataArray(obs_data, dims=["x"], name="obs")
    mod = xr.DataArray(mod_data, dims=["x"], name="mod")

    plot_obj = ConditionalBiasPlot(data=mod)
    ax = plot_obj.plot(obs=obs, n_bins=5)
    assert ax is not None
    assert plot_obj.ax.get_xlabel() == "Observed Value"


def test_conditional_bias_plot_dataarray_missing_obs(clear_figures):
    """Verify ConditionalBiasPlot raises error when obs is missing for DataArray."""
    mod = xr.DataArray(np.random.rand(100), dims=["x"], name="mod")
    plot_obj = ConditionalBiasPlot(data=mod)
    with pytest.raises(ValueError, match="obs must be provided"):
        plot_obj.plot(n_bins=5)


def test_conditional_bias_plot_unsupported_type(clear_figures):
    """Verify ConditionalBiasPlot raises error for unsupported data types."""
    with pytest.raises(TypeError, match="Unsupported data type"):
        ConditionalBiasPlot(data=[1, 2, 3])


def test_conditional_bias_hvplot_grouping():
    """Verify hvplot method handles grouping with label_col."""
    pytest.importorskip("holoviews")
    pytest.importorskip("hvplot")

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
    hv_plot = plot_obj.hvplot(
        obs_col="obs", fcst_col="mod", label_col="label", n_bins=5
    )

    import holoviews as hv

    assert isinstance(hv_plot, hv.core.overlay.Overlay)


def test_conditional_bias_hvplot_dataarray():
    """Verify hvplot method handles DataArray input."""
    pytest.importorskip("holoviews")
    pytest.importorskip("hvplot")

    obs_data = np.random.rand(100)
    mod_data = obs_data + 0.5
    obs = xr.DataArray(obs_data, dims=["x"], name="obs")
    mod = xr.DataArray(mod_data, dims=["x"], name="mod")

    plot_obj = ConditionalBiasPlot(data=mod)
    hv_plot = plot_obj.hvplot(obs=obs, n_bins=5)

    import holoviews as hv

    assert isinstance(hv_plot, hv.core.overlay.Overlay)


def test_conditional_bias_long_name(clear_figures):
    """Verify ConditionalBiasPlot uses long_name attribute for labels."""
    obs_data = np.random.rand(100)
    mod_data = obs_data + 0.5
    obs = xr.DataArray(
        obs_data, dims=["x"], name="obs", attrs={"long_name": "Custom Obs Label"}
    )
    mod = xr.DataArray(mod_data, dims=["x"], name="mod")

    ds = xr.Dataset({"obs": obs, "mod": mod})
    plot_obj = ConditionalBiasPlot(data=ds)
    plot_obj.plot(obs_col="obs", fcst_col="mod")
    assert plot_obj.ax.get_xlabel() == "Custom Obs Label"


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


def test_conditional_bias_plot_lazy(clear_figures):
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


def test_conditional_bias_grouping_xr(clear_figures):
    """Verify ConditionalBiasPlot handles grouping with label_col in Xarray."""
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
