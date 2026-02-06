import pytest
import numpy as np
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
    fcast = np.array([2, 3, 4])
    assert verification_metrics.compute_bias(obs, fcast) == pytest.approx(1.0)


def test_compute_rmse():
    """Test the compute_rmse function."""
    obs = np.array([1, 2, 3])
    fcast = np.array([2, 3, 4])
    assert verification_metrics.compute_rmse(obs, fcast) == pytest.approx(1.0)


def test_compute_mae():
    """Test the compute_mae function."""
    obs = np.array([1, 2, 3])
    fcast = np.array([2, 3, 4])
    assert verification_metrics.compute_mae(obs, fcast) == pytest.approx(1.0)


def test_compute_corr():
    """Test the compute_corr function."""
    obs = np.array([1, 2, 3])
    fcast = np.array([1, 2, 3])
    assert verification_metrics.compute_corr(obs, fcast) == pytest.approx(1.0)


def test_compute_spearmanr():
    """Test the compute_spearmanr function."""
    obs = np.array([1, 2, 3])
    fcast = np.array([1, 2, 3])
    assert verification_metrics.compute_spearmanr(obs, fcast) == pytest.approx(1.0)


def test_compute_kendalltau():
    """Test the compute_kendalltau function."""
    obs = np.array([1, 2, 3])
    fcast = np.array([1, 2, 3])
    assert verification_metrics.compute_kendalltau(obs, fcast) == pytest.approx(1.0)


def test_compute_ioa():
    """Test the compute_ioa function."""
    obs = np.array([1, 2, 3])
    fcast = np.array([1.1, 1.9, 3.2])
    assert verification_metrics.compute_ioa(obs, fcast) > 0.9


def test_compute_nse():
    """Test the compute_nse function."""
    obs = np.array([1, 2, 3])
    fcast = np.array([1.1, 1.9, 3.1])
    assert verification_metrics.compute_nse(obs, fcast) > 0.9


def test_compute_kge():
    """Test the compute_kge function."""
    obs = np.array([1, 2, 3])
    fcast = np.array([1.1, 1.9, 3.1])
    assert verification_metrics.compute_kge(obs, fcast) > 0.8


def test_compute_mnb():
    """Test the compute_mnb function."""
    obs = np.array([1, 2, 3])
    fcast = np.array([1.1, 2.2, 3.3])
    assert verification_metrics.compute_mnb(obs, fcast) == pytest.approx(10.0)


def test_compute_mne():
    """Test the compute_mne function."""
    obs = np.array([1, 2, 3])
    fcast = np.array([1.1, 2.2, 3.3])
    assert verification_metrics.compute_mne(obs, fcast) == pytest.approx(10.0)


def test_compute_mape():
    """Test the compute_mape function."""
    obs = np.array([1, 2, 3])
    fcast = np.array([1.1, 2.2, 3.3])
    assert verification_metrics.compute_mape(obs, fcast) == pytest.approx(10.0)


def test_compute_mase():
    """Test the compute_mase function."""
    obs = np.array([1, 2, 3, 4])
    fcast = np.array([1.1, 2.1, 3.1, 4.1])
    # Naive error is mean(abs(diff([1,2,3,4]))) = mean([1,1,1]) = 1.0
    # Model error is mean([0.1, 0.1, 0.1, 0.1]) = 0.1
    # MASE = 0.1 / 1.0 = 0.1
    assert verification_metrics.compute_mase(obs, fcast) == pytest.approx(0.1)


def test_compute_wdmb():
    """Test the compute_wdmb function."""
    obs = np.array([0, 90])
    fcast = np.array([10, 100])
    assert verification_metrics.compute_wdmb(obs, fcast) == pytest.approx(10.0)


def test_compute_stdo():
    """Test the compute_stdo function."""
    obs = np.array([1, 2, 3])
    fcast = np.array([1.1, 1.9, 3.2])
    # errors are [ -0.1, 0.1, -0.2]
    # mean error is -0.0666...
    # std of errors
    assert verification_metrics.compute_stdo(obs, fcast) > 0


def test_compute_stdp():
    """Test the compute_stdp function."""
    obs = np.array([1, 2, 3])
    fcast = np.array([1.1, 1.9, 3.2])
    assert verification_metrics.compute_stdp(obs, fcast) > 0


def test_compute_r2():
    """Test the compute_r2 function."""
    obs = np.array([1, 2, 3])
    fcast = np.array([1, 2, 3])
    assert verification_metrics.compute_r2(obs, fcast) == pytest.approx(1.0)
