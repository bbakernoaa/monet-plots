import numpy as np
import xarray as xr
import dask.array as da
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
    """Test AUC with Xarray inputs."""
    x = xr.DataArray(np.sort(np.random.rand(10)), dims=["threshold"])
    y = xr.DataArray(np.sort(np.random.rand(10)), dims=["threshold"])

    auc = verification_metrics.compute_auc(x, y)

    assert isinstance(auc, xr.DataArray)
    assert "Calculated AUC" in auc.attrs["history"]

    # Correctness
    auc_eager = verification_metrics.compute_auc(x.values, y.values)
    np.testing.assert_allclose(auc.values, auc_eager)


def test_lazy_auc_dask():
    """Test that AUC remains lazy when given dask-backed xarray inputs."""
    x_data = np.sort(np.random.rand(100))
    y_data = np.sort(np.random.rand(100))

    x = xr.DataArray(da.from_array(x_data, chunks=10), dims=["threshold"])
    y = xr.DataArray(da.from_array(y_data, chunks=10), dims=["threshold"])

    auc = verification_metrics.compute_auc(x, y)

    assert hasattr(auc.data, "chunks"), "AUC should be a dask array"
    # Verify correctness
    expected = np.trapezoid(y_data, x_data)
    np.testing.assert_allclose(auc.compute(), expected)
