import dask.array as da
import numpy as np
import pytest
import xarray as xr
from monet_plots import verification_metrics


def test_compute_pod():
    """Test the compute_pod function."""
    assert verification_metrics.compute_pod(10, 5) == pytest.approx(0.6666666666666666)
    assert verification_metrics.compute_pod(0, 0) == 0


def test_compute_far():
    """Test the compute_far function."""
    assert verification_metrics.compute_far(10, 5) == pytest.approx(0.3333333333333333)
    assert verification_metrics.compute_far(0, 0) == 0


def test_compute_success_ratio():
    """Test the compute_success_ratio function."""
    assert verification_metrics.compute_success_ratio(10, 5) == pytest.approx(
        0.6666666666666666
    )
    assert verification_metrics.compute_success_ratio(0, 0) == 0


def test_compute_csi():
    """Test the compute_csi function."""
    assert verification_metrics.compute_csi(10, 5, 2) == pytest.approx(
        0.5882352941176471
    )
    assert verification_metrics.compute_csi(0, 0, 0) == 0


def test_compute_frequency_bias():
    """Test the compute_frequency_bias function."""
    assert verification_metrics.compute_frequency_bias(10, 5, 2) == pytest.approx(0.8)
    assert verification_metrics.compute_frequency_bias(0, 0, 0) == 0


def test_compute_pofd():
    """Test the compute_pofd function."""
    assert verification_metrics.compute_pofd(5, 10) == pytest.approx(0.3333333333333333)
    assert verification_metrics.compute_pofd(0, 0) == 0


def test_compute_auc():
    """Test the compute_auc function."""
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])
    assert verification_metrics.compute_auc(x, y) == pytest.approx(0.5)


def test_compute_reliability_curve():
    """Test the compute_reliability_curve function."""
    forecasts = np.array([0.1, 0.9])
    observations = np.array([0, 1])
    bin_centers, obs_freq, bin_counts = verification_metrics.compute_reliability_curve(
        forecasts, observations, n_bins=2
    )
    assert len(bin_centers) == 2
    assert len(obs_freq) == 2
    assert len(bin_counts) == 2


def test_compute_brier_score_components():
    """Test the compute_brier_score_components function."""
    forecasts = np.array([0.1, 0.9])
    observations = np.array([0, 1])
    components = verification_metrics.compute_brier_score_components(
        forecasts, observations, n_bins=2
    )
    assert "reliability" in components
    assert "resolution" in components
    assert "uncertainty" in components
    assert "brier_score" in components


def test_compute_rank_histogram():
    """Test the compute_rank_histogram function."""
    ensemble = np.random.rand(100, 10)
    observations = np.random.rand(100)
    counts = verification_metrics.compute_rank_histogram(ensemble, observations)
    assert len(counts) == 11


def test_compute_rev():
    """Test the compute_rev function."""
    cost_loss_ratios = np.linspace(0, 1, 11)
    rev = verification_metrics.compute_rev(10, 5, 2, 20, cost_loss_ratios, 0.5)
    assert len(rev) == 11


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
    """Test REV with vectorized inputs."""
    cost_loss_ratios = np.linspace(0.1, 0.9, 10)
    # REV currently takes scalars for hits/misses but vectorized cost_loss_ratios
    rev = verification_metrics.compute_rev(10, 5, 2, 20, cost_loss_ratios, 0.5)
    assert len(rev) == 10
    # REV can be negative if forecast is worse than climatology
    assert isinstance(rev, np.ndarray)


def test_brier_score_components_xarray():
    """Test compute_brier_score_components with xarray/dask."""
    forecasts = xr.DataArray(da.from_array([0.1, 0.9], chunks=1), dims=["time"])
    observations = xr.DataArray(da.from_array([0, 1], chunks=1), dims=["time"])

    components = verification_metrics.compute_brier_score_components(
        forecasts, observations
    )

    assert "reliability" in components
    assert "resolution" in components
    assert "uncertainty" in components
    assert "brier_score" in components

    # Check that they are dask-backed or at least computed correctly
    assert isinstance(
        components["reliability"], (xr.DataArray, float, np.float64, da.Array)
    )


def test_rev_xarray():
    """Test compute_rev with xarray/dask."""
    hits = xr.DataArray(da.from_array([10], chunks=1), dims=["lat"])
    misses = xr.DataArray(da.from_array([5], chunks=1), dims=["lat"])
    fa = xr.DataArray(da.from_array([2], chunks=1), dims=["lat"])
    cn = xr.DataArray(da.from_array([20], chunks=1), dims=["lat"])
    cost_loss_ratios = np.linspace(0, 1, 11)

    rev = verification_metrics.compute_rev(hits, misses, fa, cn, cost_loss_ratios, 0.5)

    assert isinstance(rev, xr.DataArray)
    assert rev.shape == (11, 1)  # (cost_loss, lat)
    assert "Calculated REV" in rev.attrs.get("history", "")


def test_auc_xarray():
    """Test compute_auc with xarray."""
    x = xr.DataArray([0.0, 1.0], dims=["bin"])
    y = xr.DataArray([0.0, 1.0], dims=["bin"])

    auc = verification_metrics.compute_auc(x, y)
    assert isinstance(auc, xr.DataArray)
    assert float(auc) == pytest.approx(0.5)
    assert "Calculated AUC" in auc.attrs.get("history", "")
