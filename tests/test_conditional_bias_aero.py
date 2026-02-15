import numpy as np
import pytest
import xarray as xr
import holoviews as hv
from monet_plots.plots.conditional_bias import ConditionalBiasPlot


@pytest.fixture
def sample_ds():
    """Create a sample dataset with eager and lazy versions."""
    n = 1000
    obs_val = np.linspace(0, 100, n)
    mod_val = obs_val + np.random.normal(5, 2, n)  # Bias of 5

    ds_eager = xr.Dataset(
        {"obs": (("x",), obs_val), "fcst": (("x",), mod_val)},
        coords={"x": np.arange(n)},
    )

    ds_lazy = ds_eager.chunk({"x": 100})

    return ds_eager, ds_lazy


def test_parity_eager_lazy(sample_ds):
    """Verify that eager and lazy paths yield identical results."""
    ds_eager, ds_lazy = sample_ds

    # Eager plot
    plot_eager = ConditionalBiasPlot(data=ds_eager)
    plot_eager.plot(n_bins=10)
    stats_eager = plot_eager.stats

    # Lazy plot
    plot_lazy = ConditionalBiasPlot(data=ds_lazy)
    # Check that it's still lazy before plotting
    assert hasattr(plot_lazy.data.obs.data, "chunks")

    plot_lazy.plot(n_bins=10)
    stats_lazy = plot_lazy.stats

    # Verify parity
    xr.testing.assert_allclose(stats_eager, stats_lazy)

    # Verify bias value (should be around 5)
    assert np.allclose(stats_eager["mean"].mean(), 5, atol=0.5)


def test_provenance_tracking(sample_ds):
    """Verify that history attributes are updated."""
    ds_eager, _ = sample_ds
    plot = ConditionalBiasPlot(data=ds_eager)
    plot.plot(n_bins=5)

    assert "ConditionalBiasPlot" in plot.data.attrs["history"]
    assert "Computed binned bias" in plot.stats.attrs["history"]


def test_hvplot_output(sample_ds):
    """Verify that hvplot returns a HoloViews object."""
    ds_eager, _ = sample_ds
    plot = ConditionalBiasPlot(data=ds_eager)
    hv_obj = plot.hvplot(n_bins=5)

    # Should be an Overlay (HLine * ErrorBars)
    assert isinstance(hv_obj, hv.core.Overlay)
    # Check that sub-elements are there
    assert any(isinstance(el, hv.HLine) for el in hv_obj)


def test_label_col_support():
    """Test multiple groups via label_col."""
    n = 100
    obs = np.random.rand(n)
    fcst = obs + 1
    labels = np.array(["A"] * 50 + ["B"] * 50)

    ds = xr.Dataset(
        {"obs": (("x",), obs), "fcst": (("x",), fcst), "model": (("x",), labels)}
    )

    plot = ConditionalBiasPlot(data=ds)
    plot.plot(label_col="model", n_bins=5)

    # Stats should have 'model' dimension
    assert "model" in plot.stats.dims
    assert len(plot.stats.model) == 2
    assert "A" in plot.stats.model.values


def test_min_samples_filtering():
    """Verify that bins with few samples are filtered out."""
    # Only 2 samples, but n_bins=10 and min_samples=5
    obs = np.array([1.0, 10.0])
    fcst = np.array([1.1, 10.1])
    ds = xr.Dataset({"obs": (("x",), obs), "fcst": (("x",), fcst)})

    plot = ConditionalBiasPlot(data=ds)
    plot.plot(n_bins=10, min_samples=5)

    # All bins should be filtered out in the plot (errorbars)
    # but the stats object itself still contains the raw calculated values (with NaNs or low counts)
    # We check the Matplotlib lines
    lines = plot.ax.get_lines()
    # Only the zero-bias line should remain
    assert len(lines) == 1
