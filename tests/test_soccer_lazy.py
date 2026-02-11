import numpy as np
import pandas as pd
import pytest
import xarray as xr

try:
    import dask.array as da
except ImportError:
    da = None

from monet_plots.plots.soccer import SoccerPlot


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    obs = np.array([10, 20, 30, 40, 50])
    mod = np.array([12, 18, 35, 38, 55])
    labels = ["A", "B", "C", "D", "E"]
    return obs, mod, labels


def test_soccer_eager_pandas(sample_data):
    """Test SoccerPlot with pandas DataFrame (Eager)."""
    obs, mod, labels = sample_data
    df = pd.DataFrame({"obs": obs, "mod": mod, "label": labels})

    plot = SoccerPlot(df, obs_col="obs", mod_col="mod", label_col="label")
    assert isinstance(plot.data, pd.DataFrame)

    # Check calculated metrics
    # Fractional Bias: 200 * (mod - obs) / (mod + obs)
    expected_bias = 200.0 * (mod - obs) / (mod + obs)
    np.testing.assert_allclose(plot.bias_data, expected_bias)

    ax = plot.plot()
    assert ax is not None
    plot.close()


@pytest.mark.skipif(da is None, reason="dask not installed")
def test_soccer_lazy_xarray(sample_data):
    """Test SoccerPlot with dask-backed xarray (Lazy)."""
    obs, mod, labels = sample_data
    ds = xr.Dataset(
        {
            "obs": (["x"], obs),
            "mod": (["x"], mod),
            "label": (["x"], labels),
        },
        coords={"x": np.arange(len(obs))},
    ).chunk({"x": 2})

    plot = SoccerPlot(ds, obs_col="obs", mod_col="mod", label_col="label")

    # Verify it's still lazy
    assert isinstance(plot.bias_data.data, da.Array)
    assert "history" in plot.bias_data.attrs
    assert "Calculated Mean Fractional Bias" in plot.bias_data.attrs["history"]
    assert "(monet-plots)" in plot.bias_data.attrs["history"]

    # Trigger plot (which calls compute)
    ax = plot.plot()
    assert ax is not None

    # Verify values
    expected_bias = 200.0 * (mod - obs) / (mod + obs)
    np.testing.assert_allclose(plot.bias_data.compute(), expected_bias)

    plot.close()


def test_soccer_normalized_metric(sample_data):
    """Test SoccerPlot with normalized metric."""
    obs, mod, labels = sample_data
    df = pd.DataFrame({"obs": obs, "mod": mod})

    plot = SoccerPlot(df, obs_col="obs", mod_col="mod", metric="normalized")

    # Normalized Bias: 100 * (mod - obs) / obs
    expected_bias = 100.0 * (mod - obs) / obs
    np.testing.assert_allclose(plot.bias_data, expected_bias)
    plot.close()


def test_soccer_precalculated(sample_data):
    """Test SoccerPlot with pre-calculated bias and error."""
    obs, mod, labels = sample_data
    bias = mod - obs
    error = np.abs(mod - obs)
    df = pd.DataFrame({"bias": bias, "error": error})

    plot = SoccerPlot(df, bias_col="bias", error_col="error")
    np.testing.assert_allclose(plot.bias_data, bias)
    np.testing.assert_allclose(plot.error_data, error)
    plot.close()
