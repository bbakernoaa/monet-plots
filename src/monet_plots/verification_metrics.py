import numpy as np
from typing import Tuple, Union, Dict


def compute_pod(hits: Union[int, np.ndarray], misses: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculates Probability of Detection (POD) or Hit Rate.

    POD = Hits / (Hits + Misses)
    """
    denominator = hits + misses
    return np.divide(
        hits,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator != 0,
    )


def compute_far(hits: Union[int, np.ndarray], fa: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculates False Alarm Ratio (FAR).

    FAR = False Alarms / (Hits + False Alarms)
    """
    denominator = hits + fa
    return np.divide(
        fa,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator != 0,
    )


def compute_success_ratio(hits: Union[int, np.ndarray], fa: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculates Success Ratio (SR).

    SR = 1 - FAR = Hits / (Hits + False Alarms)
    """
    denominator = hits + fa
    return np.divide(
        hits,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator != 0,
    )


def compute_csi(
    hits: Union[int, np.ndarray],
    misses: Union[int, np.ndarray],
    fa: Union[int, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Calculates Critical Success Index (CSI).

    CSI = Hits / (Hits + Misses + False Alarms)
    """
    denominator = hits + misses + fa
    return np.divide(
        hits,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator != 0,
    )


def compute_frequency_bias(
    hits: Union[int, np.ndarray],
    misses: Union[int, np.ndarray],
    fa: Union[int, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Calculates Frequency Bias.

    Bias = (Hits + False Alarms) / (Hits + Misses)
    """
    numerator = hits + fa
    denominator = hits + misses
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator != 0,
    )


def compute_pofd(fa: Union[int, np.ndarray], cn: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculates Probability of False Detection (POFD).

    POFD = False Alarms / (False Alarms + Correct Negatives)
    """
    denominator = fa + cn
    return np.divide(
        fa,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator != 0,
    )


def compute_auc(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculates Area Under Curve (AUC) using the trapezoidal rule.

    Args:
        x: x-coordinates (e.g., POFD)
        y: y-coordinates (e.g., POD)
    """
    # Ensure sorted by x
    sort_idx = np.argsort(x)
    return float(np.trapezoid(y[sort_idx], x[sort_idx]))


def compute_reliability_curve(
    forecasts: np.ndarray, observations: np.ndarray, n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes reliability curve statistics.

    Args:
        forecasts: Array of forecast probabilities [0, 1]
        observations: Array of binary outcomes (0 or 1)
        n_bins: Number of bins

    Returns:
        Tuple of (bin_centers, observed_frequencies, bin_counts)
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Digitize returns indices of bins
    bin_indices = np.digitize(forecasts, bins) - 1

    # Adjust for values exactly at 1.0 (put in last bin)
    bin_indices[bin_indices == n_bins] = n_bins - 1

    observed_frequencies = []
    bin_counts = []

    for i in range(n_bins):
        mask = bin_indices == i
        count = np.sum(mask)
        bin_counts.append(count)

        if count > 0:
            observed_frequencies.append(np.mean(observations[mask]))
        else:
            observed_frequencies.append(np.nan)

    return bin_centers, np.array(observed_frequencies), np.array(bin_counts)


def compute_brier_score_components(forecasts: np.ndarray, observations: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
    """
    Decomposes Brier Score into Reliability, Resolution, and Uncertainty.

    BS = Reliability - Resolution + Uncertainty
    """
    N = len(forecasts)
    base_rate = float(np.mean(observations))
    uncertainty = base_rate * (1.0 - base_rate)

    bin_centers, obs_freq, bin_counts = compute_reliability_curve(forecasts, observations, n_bins)

    # Filter out empty bins
    mask = ~np.isnan(obs_freq)
    bin_centers = bin_centers[mask]
    obs_freq = obs_freq[mask]
    bin_counts = bin_counts[mask]

    # Reliability: Weighted average of (forecast - observed_freq)^2
    reliability = float(np.sum(bin_counts * (bin_centers - obs_freq) ** 2) / N)

    # Resolution: Weighted average of (observed_freq - base_rate)**2
    resolution = float(np.sum(bin_counts * (obs_freq - base_rate) ** 2) / N)

    return {
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": float(uncertainty),
        "brier_score": float(reliability - resolution + uncertainty),
    }


def compute_rank_histogram(ensemble: np.ndarray, observations: np.ndarray) -> np.ndarray:
    """
    Computes rank histogram counts.

    Args:
        ensemble: Shape (n_samples, n_members)
        observations: Shape (n_samples,)

    Returns:
        Array of counts for each rank (length n_members + 1)
    """
    n_samples, n_members = ensemble.shape
    ranks = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        # Count how many ensemble members are less than observation
        # Ties handling: random or specific logic? Standard is usually <
        # Here we implement standard count of members < observation
        ranks[i] = np.sum(ensemble[i] < observations[i])

    counts = np.bincount(ranks, minlength=n_members + 1)
    return counts


def compute_rev(
    hits: float,
    misses: float,
    fa: float,
    cn: float,
    cost_loss_ratios: np.ndarray,
    climatology: float,
) -> np.ndarray:
    """
    Calculates Relative Economic Value (REV).

    REV = (E_clim - E_forecast) / (E_clim - E_perfect)

    Where E is expected expense per event.
    """
    # Total N
    n = hits + misses + fa + cn

    # Probabilities
    # p_hit = hits / n
    # p_miss = misses / n
    # p_fa = fa / n
    # p_cn = cn / n

    # Alternatively, use sample base rate if climatology not provided,
    # but usually climatology is external or sample-based.
    # Here we assume the contingency table reflects the performance at a specific threshold.
    # Ideally, for a curve, we need hits/misses/fa/cn at EACH threshold corresponding to the optimal decision for a given C/L.
    # But often REV is calculated for a fixed system against varying users (C/L).

    # Expense Forecast = Cost * (Hits + False Alarms) + Loss * Misses
    # Normalize by N: E_f = C * (H+FA)/N + L * M/N
    # Let alpha = Cost/Loss ratio. Then normalized expense E'_f = alpha * (H+FA)/N + M/N

    rev_values = []

    # Base rate from sample
    s = (hits + misses) / n

    for alpha in cost_loss_ratios:
        # Expected Expense for Forecast
        # Expense = Cost * (False Alarms + Hits) + Loss * Misses
        # We divide by Loss * N to get normalized expense
        # E_norm = alpha * (FA + Hits)/N + Misses/N

        e_fcst = alpha * (hits + fa) / n + misses / n

        # Expected Expense for Climatology
        # If alpha < s: Always Protect. Expense = Cost. Norm = alpha.
        # If alpha >= s: Never Protect. Expense = Loss * s * N. Norm = s.
        e_clim = min(alpha, s)

        # Expected Expense for Perfect Forecast
        # Protect only when event occurs. Expense = Cost * s * N. Norm = alpha * s.
        e_perf = alpha * s

        if e_clim == e_perf:
            rev = 0.0  # Avoid division by zero, though usually means alpha=s or s=0/1
        else:
            rev = (e_clim - e_fcst) / (e_clim - e_perf)

        rev_values.append(rev)

    return np.array(rev_values)
