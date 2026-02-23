import numpy as np
import pytest

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


def test_compute_bias():
    """Test the compute_bias function."""
    obs = np.array([1, 2, 3])
    mod = np.array([2, 4, 6])
    assert verification_metrics.compute_bias(obs, mod) == pytest.approx(2.0)


def test_compute_rmse():
    """Test the compute_rmse function."""
    obs = np.array([1, 2, 3])
    mod = np.array([2, 4, 6])
    # squared diffs: (1-2)^2=1, (2-4)^2=4, (3-6)^2=9 -> mean=14/3 -> sqrt(14/3)
    assert verification_metrics.compute_rmse(obs, mod) == pytest.approx(np.sqrt(14 / 3))


def test_compute_mae():
    """Test the compute_mae function."""
    obs = np.array([1, 2, 3])
    mod = np.array([2, 4, 6])
    # abs diffs: 1, 2, 3 -> mean=2.0
    assert verification_metrics.compute_mae(obs, mod) == pytest.approx(2.0)


def test_compute_corr():
    """Test the compute_corr function."""
    obs = np.array([1, 2, 3])
    mod = np.array([2, 4, 6])
    assert verification_metrics.compute_corr(obs, mod) == pytest.approx(1.0)
