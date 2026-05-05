import dask.array as da
import numpy as np
import xarray as xr

from monet_plots import verification_metrics


def test_lazy_pod():
    """Test POD with lazy xarray/dask inputs."""
    hits = np.array([10, 0, 5])
    misses = np.array([5, 0, 2])

    hits_xr = xr.DataArray(hits, dims=["x"], name="hits")
    misses_xr = xr.DataArray(misses, dims=["x"], name="misses")

    # Eager
    res_eager = verification_metrics.compute_pod(hits_xr, misses_xr)

    # Lazy
    hits_lazy = hits_xr.chunk({"x": 1})
    misses_lazy = misses_xr.chunk({"x": 1})
    res_lazy = verification_metrics.compute_pod(hits_lazy, misses_lazy)

    assert res_lazy.chunks is not None
    xr.testing.assert_allclose(res_eager, res_lazy.compute())
    assert "Calculated POD" in res_lazy.attrs["history"]


def test_lazy_reliability_curve():
    """Test reliability curve with dask inputs."""
    forecasts = np.random.rand(100)
    observations = np.random.randint(0, 2, 100)

    # Eager
    bc1, of1, ct1 = verification_metrics.compute_reliability_curve(
        forecasts, observations, n_bins=5
    )

    # Lazy
    f_lazy = da.from_array(forecasts, chunks=20)
    o_lazy = da.from_array(observations, chunks=20)
    bc2, of2, ct2 = verification_metrics.compute_reliability_curve(
        f_lazy, o_lazy, n_bins=5
    )

    # Check if they are dask arrays
    assert hasattr(of2, "compute")
    assert hasattr(ct2, "compute")

    np.testing.assert_allclose(bc1, bc2)
    np.testing.assert_allclose(of1, of2.compute())
    np.testing.assert_allclose(ct1, ct2.compute())


def test_lazy_rank_histogram():
    """Test rank histogram with dask inputs."""
    ensemble = np.random.rand(100, 10)
    observations = np.random.rand(100)

    # Eager
    counts1 = verification_metrics.compute_rank_histogram(ensemble, observations)

    # Lazy
    e_lazy = da.from_array(ensemble, chunks=(20, 10))
    o_lazy = da.from_array(observations, chunks=20)
    counts2 = verification_metrics.compute_rank_histogram(e_lazy, o_lazy)

    assert hasattr(counts2, "compute")
    np.testing.assert_allclose(counts1, counts2.compute())


def test_lazy_rev():
    """Test REV with vectorized and lazy inputs."""
    cost_loss_ratios = np.linspace(0.1, 0.9, 10)

    # Xarray spatial inputs
    shape = (5, 5)
    hits = xr.DataArray(
        da.from_array(np.random.randint(0, 10, shape), chunks=2), dims=["x", "y"]
    )
    misses = xr.DataArray(
        da.from_array(np.random.randint(0, 10, shape), chunks=2), dims=["x", "y"]
    )
    fa = xr.DataArray(
        da.from_array(np.random.randint(0, 10, shape), chunks=2), dims=["x", "y"]
    )
    cn = xr.DataArray(
        da.from_array(np.random.randint(0, 10, shape), chunks=2), dims=["x", "y"]
    )

    rev = verification_metrics.compute_rev(hits, misses, fa, cn, cost_loss_ratios)

    assert isinstance(rev, xr.DataArray)
    assert rev.chunks is not None
    assert "cost_loss_ratio" in rev.dims
    assert rev.shape == (10, 5, 5)
    assert "Calculated Relative Economic Value" in rev.attrs["history"]


def test_lazy_brier_score_components():
    """Test Brier Score decomposition with lazy multidimensional inputs."""
    shape = (10, 10)
    forecasts = np.random.rand(*shape)
    observations = np.random.randint(0, 2, shape)

    f_lazy = xr.DataArray(da.from_array(forecasts, chunks=5), dims=["x", "y"])
    o_lazy = xr.DataArray(da.from_array(observations, chunks=5), dims=["x", "y"])

    res = verification_metrics.compute_brier_score_components(f_lazy, o_lazy, n_bins=5)

    assert isinstance(res["reliability"], xr.DataArray)
    assert res["reliability"].chunks is not None

    # Verify correctness against eager
    res_eager = verification_metrics.compute_brier_score_components(
        forecasts.flatten(), observations.flatten(), n_bins=5
    )
    np.testing.assert_allclose(res["reliability"].compute(), res_eager["reliability"])
    np.testing.assert_allclose(res["brier_score"].compute(), res_eager["brier_score"])
    assert "Computed Brier Score component" in res["reliability"].attrs["history"]


def test_lazy_auc():
    """Robust test for AUC with Dask-backed xarray inputs, including multidimensional."""
    # 1D Case
    x_data = np.sort(np.random.rand(10))
    y_data = np.random.rand(10)

    x_lazy = xr.DataArray(da.from_array(x_data, chunks=5), dims=["threshold"])
    y_lazy = xr.DataArray(da.from_array(y_data, chunks=5), dims=["threshold"])

    auc_lazy = verification_metrics.compute_auc(x_lazy, y_lazy)

    assert auc_lazy.chunks is not None
    assert "Calculated AUC" in auc_lazy.attrs["history"]

    # Eager comparison
    auc_eager = verification_metrics.compute_auc(x_data, y_data)
    np.testing.assert_allclose(auc_lazy.compute(), auc_eager)

    # Multidimensional Case
    shape = (5, 5, 10)  # lat, lon, threshold
    x_multi = np.broadcast_to(x_data, shape).copy()
    y_multi = np.random.rand(*shape)

    x_multi_lazy = xr.DataArray(
        da.from_array(x_multi, chunks=(5, 5, 5)), dims=["x", "y", "threshold"]
    )
    y_multi_lazy = xr.DataArray(
        da.from_array(y_multi, chunks=(5, 5, 5)), dims=["x", "y", "threshold"]
    )

    auc_multi_lazy = verification_metrics.compute_auc(
        x_multi_lazy, y_multi_lazy, dim="threshold"
    )

    assert auc_multi_lazy.chunks is not None
    assert auc_multi_lazy.dims == ("x", "y")
    assert auc_multi_lazy.shape == (5, 5)

    # Verify correctness for one pixel
    auc_pixel_lazy = auc_multi_lazy.isel(x=0, y=0).compute()
    auc_pixel_eager = verification_metrics.compute_auc(
        x_multi[0, 0, :], y_multi[0, 0, :]
    )
    np.testing.assert_allclose(auc_pixel_lazy, auc_pixel_eager)


def test_lazy_bias_rmse_mae():
    """Test Bias, RMSE, and MAE with lazy xarray inputs."""
    obs_data = np.random.rand(10, 10)
    mod_data = obs_data + 0.1

    obs_lazy = xr.DataArray(obs_data, dims=["x", "y"]).chunk({"x": 5})
    mod_lazy = xr.DataArray(mod_data, dims=["x", "y"]).chunk({"x": 5})

    bias = verification_metrics.compute_bias(obs_lazy, mod_lazy)
    rmse = verification_metrics.compute_rmse(obs_lazy, mod_lazy)
    mae = verification_metrics.compute_mae(obs_lazy, mod_lazy)

    assert bias.chunks is not None
    assert rmse.chunks is not None
    assert mae.chunks is not None

    np.testing.assert_allclose(bias.compute(), 0.1)
    np.testing.assert_allclose(rmse.compute(), 0.1)
    np.testing.assert_allclose(mae.compute(), 0.1)

    assert "Calculated Mean Bias" in bias.attrs["history"]


def test_multidim_rank_histogram():
    """Test rank histogram with multidimensional dimension-aware inputs."""
    shape = (5, 5, 10)  # lat, lon, member
    ensemble_data = np.random.rand(*shape)
    obs_data = np.random.rand(5, 5)

    ens_xr = xr.DataArray(
        ensemble_data, dims=["lat", "lon", "member"], name="ensemble"
    ).chunk({"lat": 2})
    obs_xr = xr.DataArray(obs_data, dims=["lat", "lon"], name="obs").chunk({"lat": 2})

    counts = verification_metrics.compute_rank_histogram(
        ens_xr, obs_xr, member_dim="member"
    )

    assert isinstance(counts, xr.DataArray)
    assert counts.chunks is not None
    assert counts.shape == (11,)
    assert counts.sum().compute() == 25  # 5*5 samples
    assert "dimension-aware" in counts.attrs["history"]


def test_lazy_crps():
    """Test CRPS with lazy multidimensional inputs."""
    shape = (5, 5, 10)  # lat, lon, member
    ensemble_data = np.random.rand(*shape)
    obs_data = np.random.rand(5, 5)

    # Eager calculation for reference
    ens_pixel = ensemble_data[0, 0, :]
    obs_pixel = obs_data[0, 0]

    # Manual CRPS for one pixel (O(M^2) for verification)
    m = len(ens_pixel)
    mae = np.mean(np.abs(ens_pixel - obs_pixel))
    diffs = np.abs(ens_pixel[:, None] - ens_pixel[None, :])
    spread = np.sum(diffs) / (2 * m * m)
    expected_pixel = mae - spread

    ens_xr = xr.DataArray(
        da.from_array(ensemble_data, chunks=(5, 5, 10)),
        dims=["lat", "lon", "member"],
        name="ensemble",
    )
    obs_xr = xr.DataArray(
        da.from_array(obs_data, chunks=(5, 5)), dims=["lat", "lon"], name="obs"
    )

    crps_lazy = verification_metrics.compute_crps(ens_xr, obs_xr, member_dim="member")

    assert crps_lazy.chunks is not None
    assert crps_lazy.dims == ("lat", "lon")

    # Check pixel value
    np.testing.assert_allclose(crps_lazy.compute()[0, 0], expected_pixel)
    assert "Calculated CRPS" in crps_lazy.attrs["history"]
