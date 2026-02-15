import numpy as np
import pandas as pd
from typing import Tuple, Union, Dict, Any

try:
    import monet_stats
except ImportError:
    monet_stats = None

# Optional xarray import - will be used if available
try:
    import xarray as xr
except ImportError:
    xr = None


def _update_history(obj: Any, msg: str) -> Any:
    """
    Update the 'history' attribute of an xarray object to track provenance.
    """
    if xr is not None and isinstance(obj, (xr.DataArray, xr.Dataset)):
        history = obj.attrs.get("history", "")
        new_history = f"{msg} (monet-plots); {history}".strip("; ")
        obj.attrs["history"] = new_history
    return obj


def _is_xarray(obj: Any) -> bool:
    """Check if object is an xarray DataArray or Dataset."""
    return xr is not None and isinstance(obj, (xr.DataArray, xr.Dataset))


def _mean(obj: Any, dim: Any) -> Any:
    """Polymorphic mean that handles xarray 'dim' and numpy 'axis'."""
    if _is_xarray(obj):
        return obj.mean(dim=dim)
    # Convert pandas to numpy to support axis=()
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        obj = np.asarray(obj)
    return np.mean(obj, axis=dim)


def _sum(obj: Any, dim: Any) -> Any:
    """Polymorphic sum that handles xarray 'dim' and numpy 'axis'."""
    if _is_xarray(obj):
        return obj.sum(dim=dim)
    # Convert pandas to numpy to support axis=()
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        obj = np.asarray(obj)
    return np.sum(obj, axis=dim)


def compute_mb(obs: Any, mod: Any, dim: Any = None) -> Any:
    """Mean Bias (MB)."""
    if monet_stats is None:
        diff = mod - obs
        res = _mean(diff, dim)
        return _update_history(res, "Computed MB")
    # Convert pandas to numpy for monet-stats compatibility with axis=()
    if isinstance(obs, (pd.Series, pd.DataFrame)):
        obs = np.asarray(obs)
    if isinstance(mod, (pd.Series, pd.DataFrame)):
        mod = np.asarray(mod)
    # monet-stats.MB returns (obs - mod), so we swap arguments to get (mod - obs)
    res = monet_stats.MB(mod, obs, axis=dim)
    return _update_history(res, "Computed MB")


def compute_rmse(obs: Any, mod: Any, dim: Any = None) -> Any:
    """Root Mean Square Error (RMSE)."""
    if monet_stats is None:
        diff_sq = (mod - obs) ** 2
        res = np.sqrt(_mean(diff_sq, dim))
        return _update_history(res, "Computed RMSE")
    # Convert pandas to numpy for monet-stats compatibility with axis=()
    if isinstance(obs, (pd.Series, pd.DataFrame)):
        obs = np.asarray(obs)
    if isinstance(mod, (pd.Series, pd.DataFrame)):
        mod = np.asarray(mod)
    res = monet_stats.RMSE(obs, mod, axis=dim)
    return _update_history(res, "Computed RMSE")


def compute_mae(obs: Any, mod: Any, dim: Any = None) -> Any:
    """Mean Absolute Error (MAE)."""
    if monet_stats is None:
        abs_diff = np.abs(mod - obs)
        res = _mean(abs_diff, dim)
        return _update_history(res, "Computed MAE")
    # Convert pandas to numpy for monet-stats compatibility with axis=()
    if isinstance(obs, (pd.Series, pd.DataFrame)):
        obs = np.asarray(obs)
    if isinstance(mod, (pd.Series, pd.DataFrame)):
        mod = np.asarray(mod)
    res = monet_stats.MAE(obs, mod, axis=dim)
    return _update_history(res, "Computed MAE")


def compute_correlation(obs: Any, mod: Any, dim: Any = None) -> Any:
    """Pearson Correlation Coefficient."""
    if monet_stats is None:
        # Simplified fallback for NumPy/Xarray
        o = obs.values if hasattr(obs, "values") else obs
        m = mod.values if hasattr(mod, "values") else mod
        return np.corrcoef(o.ravel(), m.ravel())[0, 1]
    # Convert pandas to numpy for monet-stats compatibility with axis=()
    if isinstance(obs, (pd.Series, pd.DataFrame)):
        obs = np.asarray(obs)
    if isinstance(mod, (pd.Series, pd.DataFrame)):
        mod = np.asarray(mod)
    res = monet_stats.pearsonr(obs, mod, axis=dim)
    return _update_history(res, "Computed Correlation")


def compute_fb(obs: Any, mod: Any, dim: Any = None) -> Any:
    """Fractional Bias (FB)."""
    if monet_stats is None:
        term = (mod - obs) / (mod + obs)
        res = 200.0 * _mean(term, dim)
        return _update_history(res, "Computed FB")
    # Convert pandas to numpy for monet-stats compatibility with axis=()
    if isinstance(obs, (pd.Series, pd.DataFrame)):
        obs = np.asarray(obs)
    if isinstance(mod, (pd.Series, pd.DataFrame)):
        mod = np.asarray(mod)
    res = monet_stats.FB(obs, mod, axis=dim)
    return _update_history(res, "Computed FB")


def compute_fe(obs: Any, mod: Any, dim: Any = None) -> Any:
    """Fractional Error (FE)."""
    if monet_stats is None:
        term = np.abs(mod - obs) / (mod + obs)
        res = 200.0 * _mean(term, dim)
        return _update_history(res, "Computed FE")
    # Convert pandas to numpy for monet-stats compatibility with axis=()
    if isinstance(obs, (pd.Series, pd.DataFrame)):
        obs = np.asarray(obs)
    if isinstance(mod, (pd.Series, pd.DataFrame)):
        mod = np.asarray(mod)
    res = monet_stats.FE(obs, mod, axis=dim)
    return _update_history(res, "Computed FE")


def compute_nmb(obs: Any, mod: Any, dim: Any = None) -> Any:
    """Normalized Mean Bias (NMB)."""
    if monet_stats is None:
        diff = mod - obs
        num = _sum(diff, dim)
        den = _sum(obs, dim)
        res = 100.0 * num / den
        return _update_history(res, "Computed NMB")
    # Convert pandas to numpy for monet-stats compatibility with axis=()
    if isinstance(obs, (pd.Series, pd.DataFrame)):
        obs = np.asarray(obs)
    if isinstance(mod, (pd.Series, pd.DataFrame)):
        mod = np.asarray(mod)
    res = monet_stats.NMB(obs, mod, axis=dim)
    return _update_history(res, "Computed NMB")


def compute_nme(obs: Any, mod: Any, dim: Any = None) -> Any:
    """Normalized Mean Error (NME)."""
    if monet_stats is None:
        abs_diff = np.abs(mod - obs)
        num = _sum(abs_diff, dim)
        den = _sum(obs, dim)
        res = 100.0 * num / den
        return _update_history(res, "Computed NME")
    # Convert pandas to numpy for monet-stats compatibility with axis=()
    if isinstance(obs, (pd.Series, pd.DataFrame)):
        obs = np.asarray(obs)
    if isinstance(mod, (pd.Series, pd.DataFrame)):
        mod = np.asarray(mod)
    # monet-stats uses MNE for Mean Normalized Error (Gross Error)
    res = monet_stats.MNE(obs, mod, axis=dim)
    return _update_history(res, "Computed NME")


def compute_pod(
    hits: Union[int, np.ndarray], misses: Union[int, np.ndarray]
) -> Union[float, np.ndarray]:
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


def compute_far(
    hits: Union[int, np.ndarray], fa: Union[int, np.ndarray]
) -> Union[float, np.ndarray]:
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


def compute_success_ratio(
    hits: Union[int, np.ndarray], fa: Union[int, np.ndarray]
) -> Union[float, np.ndarray]:
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


def compute_pofd(
    fa: Union[int, np.ndarray], cn: Union[int, np.ndarray]
) -> Union[float, np.ndarray]:
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
    forecasts: Any, observations: Any, n_bins: int = 10, dim: Any = None
) -> Tuple[np.ndarray, Any, Any]:
    """
    Computes reliability curve statistics, supporting lazy evaluation and multidimensional input.

    Args:
        forecasts: Forecast probabilities [0, 1].
        observations: Binary outcomes (0 or 1).
        n_bins: Number of bins.
        dim: Dimension(s) to aggregate over. If None, aggregates over all dimensions.

    Returns:
        Tuple of (bin_centers, observed_frequencies, bin_counts)
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    if xr is not None and isinstance(forecasts, (xr.DataArray, xr.Dataset)):
        if dim is None:
            dim = list(forecasts.dims)
        elif isinstance(dim, str):
            dim = [dim]

        def _core_reliability(f, o):
            indices = np.digitize(f, bins) - 1
            indices[indices == n_bins] = n_bins - 1

            freq = np.full(n_bins, np.nan)
            counts = np.zeros(n_bins)
            for i in range(n_bins):
                mask = indices == i
                c = np.sum(mask)
                counts[i] = c
                if c > 0:
                    freq[i] = np.mean(o[mask])
            return freq, counts

        # Ensure core dimensions are unchunked for apply_ufunc
        forecasts = forecasts.chunk({d: -1 for d in dim})
        observations = observations.chunk({d: -1 for d in dim})

        res_freq, res_counts = xr.apply_ufunc(
            _core_reliability,
            forecasts,
            observations,
            input_core_dims=[dim, dim],
            output_core_dims=[["bin"], ["bin"]],
            dask="parallelized",
            vectorize=True,
            output_dtypes=[float, float],
            dask_gufunc_kwargs={"output_sizes": {"bin": n_bins}},
        )
        res_freq = res_freq.assign_coords(bin=bin_centers)
        res_counts = res_counts.assign_coords(bin=bin_centers)

        return bin_centers, res_freq, res_counts
    else:
        # Numpy implementation
        bin_indices = np.digitize(forecasts, bins) - 1
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


def compute_brier_score_components(
    forecasts: Any, observations: Any, n_bins: int = 10
) -> Dict[str, float]:
    """
    Decomposes Brier Score into Reliability, Resolution, and Uncertainty.

    BS = Reliability - Resolution + Uncertainty
    """
    # Use xarray/dask aware operations for N and base_rate
    if _is_xarray(observations):
        N = observations.size
        base_rate = observations.mean()
    else:
        N = len(observations)
        base_rate = np.mean(observations)

    bin_centers, obs_freq, bin_counts = compute_reliability_curve(
        forecasts, observations, n_bins
    )

    # Use nansum and fillna to avoid eager masking
    if _is_xarray(obs_freq):
        # Reliability: Weighted average of (forecast - observed_freq)^2
        rel_term = bin_counts * (bin_centers - obs_freq.fillna(bin_centers)) ** 2
        reliability = rel_term.sum() / N

        # Resolution: Weighted average of (observed_freq - base_rate)**2
        res_term = bin_counts * (obs_freq.fillna(base_rate) - base_rate) ** 2
        resolution = res_term.sum() / N

        uncertainty = base_rate * (1.0 - base_rate)

        # Compute results in one go if they are dask objects
        if hasattr(reliability, "compute"):
            import dask

            reliability, resolution, uncertainty, base_rate = dask.compute(
                reliability, resolution, uncertainty, base_rate
            )
    else:
        # Numpy path
        mask = ~np.isnan(obs_freq)
        reliability = (
            np.sum(bin_counts[mask] * (bin_centers[mask] - obs_freq[mask]) ** 2) / N
        )
        resolution = np.sum(bin_counts[mask] * (obs_freq[mask] - base_rate) ** 2) / N
        uncertainty = base_rate * (1.0 - base_rate)

    return {
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(uncertainty),
        "brier_score": float(reliability - resolution + uncertainty),
        "base_rate": float(base_rate),
    }


def compute_rank_histogram(ensemble: Any, observations: Any) -> np.ndarray:
    """
    Computes rank histogram counts using vectorized operations.

    Args:
        ensemble: Shape (n_samples, n_members)
        observations: Shape (n_samples,)

    Returns:
        Array of counts for each rank (length n_members + 1)
    """
    # Vectorized comparison: (n_samples, n_members) < (n_samples, 1)
    if hasattr(ensemble, "values"):
        ens_vals = ensemble.values
    else:
        ens_vals = ensemble

    if hasattr(observations, "values"):
        obs_vals = observations.values
    else:
        obs_vals = observations

    if len(obs_vals.shape) == 1:
        obs_vals = obs_vals[:, np.newaxis]

    ranks = np.sum(ens_vals < obs_vals, axis=1)

    if hasattr(ranks, "compute"):
        ranks = ranks.compute()

    n_members = ensemble.shape[1]
    counts = np.bincount(ranks.astype(int), minlength=n_members + 1)
    return counts


def compute_contingency_table(
    obs: Any, mod: Any, threshold: float, dim: Any = None
) -> Dict[str, Any]:
    """
    Computes contingency table components (hits, misses, fa, cn) from raw data.

    Args:
        obs: Observations.
        mod: Model values.
        threshold: Threshold for event detection.
        dim: Dimension(s) to aggregate over.

    Returns:
        Dictionary with 'hits', 'misses', 'fa', 'cn'.
    """
    obs_event = obs >= threshold
    mod_event = mod >= threshold

    hits = _sum(obs_event & mod_event, dim)
    misses = _sum(obs_event & ~mod_event, dim)
    fa = _sum(~obs_event & mod_event, dim)
    cn = _sum(~obs_event & ~mod_event, dim)

    res = {"hits": hits, "misses": misses, "fa": fa, "cn": cn}

    for k in res:
        _update_history(res[k], f"Computed {k} at threshold {threshold}")

    return res


def compute_categorical_metrics(
    hits: Any, misses: Any, fa: Any, cn: Any
) -> Dict[str, Any]:
    """
    Computes a set of categorical metrics from contingency table components.
    """
    pod = compute_pod(hits, misses)
    far = compute_far(hits, fa)
    sr = compute_success_ratio(hits, fa)
    csi = compute_csi(hits, misses, fa)
    fbias = compute_frequency_bias(hits, misses, fa)

    return {
        "pod": pod,
        "far": far,
        "success_ratio": sr,
        "csi": csi,
        "frequency_bias": fbias,
    }


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
    # Here we assume the contingency table reflects the performance at a specific
    # threshold.
    # Ideally, for a curve, we need hits/misses/fa/cn at EACH threshold corresponding
    # to the optimal decision for a given C/L.
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
