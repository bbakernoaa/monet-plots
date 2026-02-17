import numpy as np
import pytest
import xarray as xr
from monet_plots import verification_metrics

try:
    import dask.array as da
except ImportError:
    da = None


@pytest.mark.skipif(da is None, reason="dask not installed")
def test_metrics_eager_vs_lazy_parity():
    """Verify that metrics produce identical results for numpy and dask backends."""
    # Create test data
    data_obs = np.random.rand(100) * 10
    data_mod = data_obs + np.random.randn(100)

    da_obs_eager = xr.DataArray(data_obs, dims="x", name="obs")
    da_mod_eager = xr.DataArray(data_mod, dims="x", name="mod")

    da_obs_lazy = da_obs_eager.chunk({"x": 20})
    da_mod_lazy = da_mod_eager.chunk({"x": 20})

    metrics_to_test = [
        ("mb", verification_metrics.compute_mb),
        ("rmse", verification_metrics.compute_rmse),
        ("mae", verification_metrics.compute_mae),
        ("fb", verification_metrics.compute_fb),
        ("fe", verification_metrics.compute_fe),
        ("nmb", verification_metrics.compute_nmb),
        ("nme", verification_metrics.compute_nme),
    ]

    for name, func in metrics_to_test:
        res_eager = func(da_obs_eager, da_mod_eager)
        res_lazy = func(da_obs_lazy, da_mod_lazy)

        if hasattr(res_lazy, "compute"):
            res_lazy = res_lazy.compute()

        assert np.allclose(res_eager, res_lazy), f"Parity failure for {name}"
        assert "history" in getattr(
            res_eager, "attrs", {}
        ), f"History missing for {name} (eager)"
        assert "history" in getattr(
            res_lazy, "attrs", {}
        ), f"History missing for {name} (lazy)"


@pytest.mark.skipif(da is None, reason="dask not installed")
def test_categorical_metrics_parity():
    """Verify categorical metrics parity."""
    data_obs = np.random.rand(100)
    data_mod = np.random.rand(100)
    threshold = 0.5

    da_obs_eager = xr.DataArray(data_obs, dims="x")
    da_mod_eager = xr.DataArray(data_mod, dims="x")

    da_obs_lazy = da_obs_eager.chunk({"x": 20})
    da_mod_lazy = da_mod_eager.chunk({"x": 20})

    ct_eager = verification_metrics.compute_contingency_table(
        da_obs_eager, da_mod_eager, threshold
    )
    ct_lazy = verification_metrics.compute_contingency_table(
        da_obs_lazy, da_mod_lazy, threshold
    )

    for k in ["hits", "misses", "fa", "cn"]:
        v_eager = ct_eager[k]
        v_lazy = ct_lazy[k]
        if hasattr(v_lazy, "compute"):
            v_lazy = v_lazy.compute()
        assert v_eager == v_lazy, f"Parity failure for contingency table {k}"


@pytest.mark.skipif(da is None, reason="dask not installed")
def test_reliability_curve_parity():
    """Verify reliability curve parity."""
    forecasts = np.random.rand(100)
    observations = (np.random.rand(100) > 0.5).astype(int)

    da_f_eager = xr.DataArray(forecasts, dims="x")
    da_o_eager = xr.DataArray(observations, dims="x")

    da_f_lazy = da_f_eager.chunk({"x": 20})
    da_o_lazy = da_o_eager.chunk({"x": 20})

    _, freq_eager, count_eager = verification_metrics.compute_reliability_curve(
        da_f_eager, da_o_eager
    )
    _, freq_lazy, count_lazy = verification_metrics.compute_reliability_curve(
        da_f_lazy, da_o_lazy
    )

    if hasattr(freq_lazy, "compute"):
        freq_lazy = freq_lazy.compute()
        count_lazy = count_lazy.compute()

    np.testing.assert_allclose(freq_eager, freq_lazy, equal_nan=True)
    np.testing.assert_allclose(count_eager, count_lazy)


if __name__ == "__main__":
    pytest.main([__file__])
