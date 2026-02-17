import pytest
import numpy as np
import xarray as xr
from monet_plots import verification_metrics


def test_mb_bridge():
    obs = np.array([1, 2, 3])
    mod = np.array([1.1, 1.8, 3.3])
    # MB = (0.1 - 0.2 + 0.3) / 3 = 0.2 / 3 = 0.0666...
    assert verification_metrics.compute_mb(obs, mod) == pytest.approx(
        0.06666666666666667
    )


def test_rmse_bridge():
    obs = np.array([1, 2, 3])
    mod = np.array([1.1, 1.8, 3.3])
    # RMSE = sqrt((0.1^2 + (-0.2)^2 + 0.3^2) / 3) = sqrt((0.01 + 0.04 + 0.09) / 3) = sqrt(0.14 / 3) = 0.216...
    assert verification_metrics.compute_rmse(obs, mod) == pytest.approx(
        0.21602468994692867
    )


def test_xarray_bridge():
    obs = xr.DataArray([1, 2, 3], dims="x", name="obs")
    mod = xr.DataArray([1.1, 1.8, 3.3], dims="x", name="mod")
    res = verification_metrics.compute_mb(obs, mod)
    assert isinstance(res, xr.DataArray)
    # History attribute should always be present for Xarray outputs now
    assert "history" in res.attrs
    assert "monet-plots" in res.attrs["history"]


def test_dimension_awareness():
    obs = xr.DataArray(np.ones((2, 3)), dims=["x", "y"])
    mod = xr.DataArray(np.zeros((2, 3)), dims=["x", "y"])

    # Across all dims
    res_all = verification_metrics.compute_mb(obs, mod)
    assert res_all == -1.0

    # Across dim 'x'
    res_x = verification_metrics.compute_mb(obs, mod, dim="x")
    assert res_x.shape == (3,)
    assert np.all(res_x == -1.0)


if __name__ == "__main__":
    pytest.main([__file__])
