import numpy as np
import xarray as xr
from typing import Tuple, Union, Dict, Any


def _update_history(obj: Any, msg: str) -> Any:
    """Updates the history attribute of an xarray object.

    Parameters
    ----------
    obj : Any
        The object to update (typically xarray.DataArray or xarray.Dataset).
    msg : str
        The message to add to the history.

    Returns
    -------
    Any
        The object with the updated history.
    """
    if isinstance(obj, (xr.DataArray, xr.Dataset)):
        history = obj.attrs.get("history", "")
        obj.attrs["history"] = f"{msg} (monet-plots); {history}"
    return obj


def compute_pod(
    hits: Union[int, np.ndarray, xr.DataArray],
    misses: Union[int, np.ndarray, xr.DataArray],
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculates Probability of Detection (POD) or Hit Rate.

    POD = Hits / (Hits + Misses)

    Parameters
    ----------
    hits : Union[int, np.ndarray, xr.DataArray]
        Number of hits.
    misses : Union[int, np.ndarray, xr.DataArray]
        Number of misses.

    Returns
    -------
    Union[float, np.ndarray, xr.DataArray]
        The calculated POD.
    """
    denominator = hits + misses
    if isinstance(hits, (xr.DataArray, xr.Dataset)):
        res = hits / denominator
        res = res.where(denominator != 0, 0)
        return _update_history(res, "Calculated POD")

    return np.divide(
        hits,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator != 0,
    )


def compute_far(
    hits: Union[int, np.ndarray, xr.DataArray],
    fa: Union[int, np.ndarray, xr.DataArray],
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculates False Alarm Ratio (FAR).

    FAR = False Alarms / (Hits + False Alarms)

    Parameters
    ----------
    hits : Union[int, np.ndarray, xr.DataArray]
        Number of hits.
    fa : Union[int, np.ndarray, xr.DataArray]
        Number of false alarms.

    Returns
    -------
    Union[float, np.ndarray, xr.DataArray]
        The calculated FAR.
    """
    denominator = hits + fa
    if isinstance(hits, (xr.DataArray, xr.Dataset)):
        res = fa / denominator
        res = res.where(denominator != 0, 0)
        return _update_history(res, "Calculated FAR")

    return np.divide(
        fa,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator != 0,
    )


def compute_success_ratio(
    hits: Union[int, np.ndarray, xr.DataArray],
    fa: Union[int, np.ndarray, xr.DataArray],
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculates Success Ratio (SR).

    SR = 1 - FAR = Hits / (Hits + False Alarms)

    Parameters
    ----------
    hits : Union[int, np.ndarray, xr.DataArray]
        Number of hits.
    fa : Union[int, np.ndarray, xr.DataArray]
        Number of false alarms.

    Returns
    -------
    Union[float, np.ndarray, xr.DataArray]
        The calculated Success Ratio.
    """
    denominator = hits + fa
    if isinstance(hits, (xr.DataArray, xr.Dataset)):
        res = hits / denominator
        res = res.where(denominator != 0, 0)
        return _update_history(res, "Calculated Success Ratio")

    return np.divide(
        hits,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator != 0,
    )


def compute_csi(
    hits: Union[int, np.ndarray, xr.DataArray],
    misses: Union[int, np.ndarray, xr.DataArray],
    fa: Union[int, np.ndarray, xr.DataArray],
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculates Critical Success Index (CSI).

    CSI = Hits / (Hits + Misses + False Alarms)

    Parameters
    ----------
    hits : Union[int, np.ndarray, xr.DataArray]
        Number of hits.
    misses : Union[int, np.ndarray, xr.DataArray]
        Number of misses.
    fa : Union[int, np.ndarray, xr.DataArray]
        Number of false alarms.

    Returns
    -------
    Union[float, np.ndarray, xr.DataArray]
        The calculated CSI.
    """
    denominator = hits + misses + fa
    if isinstance(hits, (xr.DataArray, xr.Dataset)):
        res = hits / denominator
        res = res.where(denominator != 0, 0)
        return _update_history(res, "Calculated CSI")

    return np.divide(
        hits,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator != 0,
    )


def compute_frequency_bias(
    hits: Union[int, np.ndarray, xr.DataArray],
    misses: Union[int, np.ndarray, xr.DataArray],
    fa: Union[int, np.ndarray, xr.DataArray],
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculates Frequency Bias.

    Bias = (Hits + False Alarms) / (Hits + Misses)

    Parameters
    ----------
    hits : Union[int, np.ndarray, xr.DataArray]
        Number of hits.
    misses : Union[int, np.ndarray, xr.DataArray]
        Number of misses.
    fa : Union[int, np.ndarray, xr.DataArray]
        Number of false alarms.

    Returns
    -------
    Union[float, np.ndarray, xr.DataArray]
        The calculated Frequency Bias.
    """
    numerator = hits + fa
    denominator = hits + misses
    if isinstance(hits, (xr.DataArray, xr.Dataset)):
        res = numerator / denominator
        res = res.where(denominator != 0, 0)
        return _update_history(res, "Calculated Frequency Bias")

    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator != 0,
    )


def compute_pofd(
    fa: Union[int, np.ndarray, xr.DataArray],
    cn: Union[int, np.ndarray, xr.DataArray],
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculates Probability of False Detection (POFD).

    POFD = False Alarms / (False Alarms + Correct Negatives)

    Parameters
    ----------
    fa : Union[int, np.ndarray, xr.DataArray]
        Number of false alarms.
    cn : Union[int, np.ndarray, xr.DataArray]
        Number of correct negatives.

    Returns
    -------
    Union[float, np.ndarray, xr.DataArray]
        The calculated POFD.
    """
    denominator = fa + cn
    if isinstance(fa, (xr.DataArray, xr.Dataset)):
        res = fa / denominator
        res = res.where(denominator != 0, 0)
        return _update_history(res, "Calculated POFD")

    return np.divide(
        fa,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator != 0,
    )


def compute_auc(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculates Area Under Curve (AUC) using the trapezoidal rule.

    Parameters
    ----------
    x : np.ndarray
        x-coordinates (e.g., POFD).
    y : np.ndarray
        y-coordinates (e.g., POD).

    Returns
    -------
    float
        The calculated AUC.
    """
    # Ensure sorted by x
    sort_idx = np.argsort(x)
    return float(np.trapezoid(y[sort_idx], x[sort_idx]))


def compute_reliability_curve(
    forecasts: Any, observations: Any, n_bins: int = 10
) -> Tuple[Any, Any, Any]:
    """
    Computes reliability curve statistics.

    Parameters
    ----------
    forecasts : Any
        Array-like of forecast probabilities [0, 1].
    observations : Any
        Array-like of binary outcomes (0 or 1).
    n_bins : int, optional
        Number of bins, by default 10.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Tuple of (bin_centers, observed_frequencies, bin_counts).
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Handle Dask for "Lazy by Default"
    is_dask = hasattr(forecasts, "chunks") or (
        isinstance(forecasts, xr.DataArray) and forecasts.chunks is not None
    )

    if is_dask:
        import dask.array as da

        f_data = forecasts.data if isinstance(forecasts, xr.DataArray) else forecasts
        o_data = (
            observations.data
            if isinstance(observations, xr.DataArray)
            else observations
        )
        bin_counts, _ = da.histogram(f_data, bins=bins)
        obs_sum, _ = da.histogram(f_data, bins=bins, weights=o_data)
    else:
        bin_counts, _ = np.histogram(forecasts, bins=bins)
        obs_sum, _ = np.histogram(forecasts, bins=bins, weights=observations)

    observed_frequencies = np.divide(
        obs_sum,
        bin_counts,
        out=np.full_like(obs_sum, np.nan, dtype=float),
        where=bin_counts > 0,
    )

    # Return as Xarray for provenance if inputs were Xarray
    if isinstance(forecasts, (xr.DataArray, xr.Dataset)):
        coords = {"bin_center": bin_centers}
        observed_frequencies = xr.DataArray(
            observed_frequencies,
            coords=coords,
            dims=["bin_center"],
            name="observed_frequency",
        )
        bin_counts = xr.DataArray(
            bin_counts, coords=coords, dims=["bin_center"], name="bin_count"
        )
        bin_centers = xr.DataArray(
            bin_centers, coords=coords, dims=["bin_center"], name="bin_center"
        )
        _update_history(observed_frequencies, "Computed reliability curve")

    return bin_centers, observed_frequencies, bin_counts


def compute_brier_score_components(
    forecasts: np.ndarray, observations: np.ndarray, n_bins: int = 10
) -> Dict[str, float]:
    """
    Decomposes Brier Score into Reliability, Resolution, and Uncertainty.

    BS = Reliability - Resolution + Uncertainty
    """
    N = len(forecasts)
    base_rate = float(np.mean(observations))
    uncertainty = base_rate * (1.0 - base_rate)

    bin_centers, obs_freq, bin_counts = compute_reliability_curve(
        forecasts, observations, n_bins
    )

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


def compute_rank_histogram(ensemble: Any, observations: Any) -> Any:
    """
    Computes rank histogram counts.

    Parameters
    ----------
    ensemble : Any
        Shape (n_samples, n_members).
    observations : Any
        Shape (n_samples,).

    Returns
    -------
    np.ndarray
        Array of counts for each rank (length n_members + 1).
    """
    # Vectorized rank computation
    # Handle Xarray/Dask
    if isinstance(ensemble, xr.DataArray):
        ensemble = ensemble.data
    if isinstance(observations, xr.DataArray):
        observations = observations.data

    # ensemble < observations[:, np.newaxis] broadcast comparison
    # This works for both numpy and dask arrays
    obs_expanded = observations[:, np.newaxis]
    ranks = (ensemble < obs_expanded).sum(axis=1)

    if hasattr(ranks, "chunks"):
        import dask.array as da

        n_members = ensemble.shape[1]
        counts, _ = da.histogram(ranks, bins=np.arange(n_members + 2) - 0.5)
    else:
        counts = np.bincount(ranks, minlength=ensemble.shape[1] + 1)

    # Return as Xarray for provenance if input was Xarray
    if isinstance(ensemble, (xr.DataArray, xr.Dataset)) or isinstance(
        observations, (xr.DataArray, xr.Dataset)
    ):
        counts = xr.DataArray(
            counts,
            coords={"rank": np.arange(len(counts))},
            dims=["rank"],
            name="rank_counts",
        )
        _update_history(counts, "Computed rank histogram")

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

    Parameters
    ----------
    hits : float
        Number of hits.
    misses : float
        Number of misses.
    fa : float
        Number of false alarms.
    cn : float
        Number of correct negatives.
    cost_loss_ratios : np.ndarray
        Array of cost/loss ratios.
    climatology : float
        Climatological base rate.

    Returns
    -------
    np.ndarray
        Array of REV values for each cost/loss ratio.
    """
    n = hits + misses + fa + cn
    alpha = np.asarray(cost_loss_ratios)
    s = (hits + misses) / n

    # Expected Expense for Forecast
    e_fcst = alpha * (hits + fa) / n + misses / n

    # Expected Expense for Climatology
    e_clim = np.minimum(alpha, s)

    # Expected Expense for Perfect Forecast
    e_perf = alpha * s

    # REV calculation
    numerator = e_clim - e_fcst
    denominator = e_clim - e_perf

    rev = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )

    return rev
