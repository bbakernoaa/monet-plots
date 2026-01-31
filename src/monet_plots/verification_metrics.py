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


def compute_auc(x: Any, y: Any) -> Union[float, xr.DataArray]:
    """
    Calculates Area Under Curve (AUC) using the trapezoidal rule.

    Parameters
    ----------
    x : Any
        x-coordinates (e.g., POFD).
    y : Any
        y-coordinates (e.g., POD).

    Returns
    -------
    Union[float, xr.DataArray]
        The calculated AUC.
    """
    if isinstance(x, (xr.DataArray, xr.Dataset)):
        # xarray integration
        # Note: np.trapezoid works on DataArrays but returns a DataArray if multiple dims
        # or a scalar DataArray.
        res = xr.DataArray(np.trapezoid(y, x))
        return _update_history(res, "Calculated AUC")

    # Ensure sorted by x for numpy
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
    Tuple[Any, Any, Any]
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
    forecasts: Any, observations: Any, n_bins: int = 10
) -> Dict[str, Any]:
    """
    Decomposes Brier Score into Reliability, Resolution, and Uncertainty.

    BS = Reliability - Resolution + Uncertainty

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
    Dict[str, Any]
        Dictionary containing 'reliability', 'resolution', 'uncertainty',
        and 'brier_score'.
    """
    if isinstance(observations, (xr.DataArray, xr.Dataset)):
        base_rate = observations.mean()
        N = observations.size
    else:
        base_rate = np.mean(observations)
        N = len(forecasts)

    uncertainty = base_rate * (1.0 - base_rate)

    bin_centers, obs_freq, bin_counts = compute_reliability_curve(
        forecasts, observations, n_bins
    )

    # Handle filtering of empty bins
    if isinstance(obs_freq, xr.DataArray):
        # Histogram results are small (n_bins), so it's safe to compute for filtering
        mask = ~obs_freq.isnull()
        if hasattr(mask.data, "chunks"):
            mask = mask.compute()

        bin_centers_f = bin_centers.where(mask, drop=True)
        obs_freq_f = obs_freq.where(mask, drop=True)
        bin_counts_f = bin_counts.where(mask, drop=True)

        reliability = (bin_counts_f * (bin_centers_f - obs_freq_f) ** 2).sum() / N
        resolution = (bin_counts_f * (obs_freq_f - base_rate) ** 2).sum() / N
    else:
        mask = ~np.isnan(obs_freq)
        bin_centers_f = bin_centers[mask]
        obs_freq_f = obs_freq[mask]
        bin_counts_f = bin_counts[mask]

        reliability = np.sum(bin_counts_f * (bin_centers_f - obs_freq_f) ** 2) / N
        resolution = np.sum(bin_counts_f * (obs_freq_f - base_rate) ** 2) / N

    return {
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "brier_score": reliability - resolution + uncertainty,
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
    Any
        Array or DataArray of counts for each rank (length n_members + 1).
    """
    # Vectorized rank computation
    # Handle Xarray/Dask
    if isinstance(ensemble, xr.DataArray):
        ensemble_data = ensemble.data
    else:
        ensemble_data = ensemble

    if isinstance(observations, xr.DataArray):
        observations_data = observations.data
    else:
        observations_data = observations

    # ensemble < observations[:, np.newaxis] broadcast comparison
    # This works for both numpy and dask arrays
    obs_expanded = observations_data[:, np.newaxis]
    ranks = (ensemble_data < obs_expanded).sum(axis=1)

    if hasattr(ranks, "chunks"):
        import dask.array as da

        n_members = ensemble_data.shape[1]
        counts, _ = da.histogram(ranks, bins=np.arange(n_members + 2) - 0.5)
    else:
        counts = np.bincount(ranks, minlength=ensemble_data.shape[1] + 1)

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
    hits: Any,
    misses: Any,
    fa: Any,
    cn: Any,
    cost_loss_ratios: Any,
    climatology: Any,
) -> Any:
    """
    Calculates Relative Economic Value (REV).

    REV = (E_clim - E_forecast) / (E_clim - E_perfect)

    Where E is expected expense per event.

    Parameters
    ----------
    hits : Any
        Number of hits (scalar or array).
    misses : Any
        Number of misses (scalar or array).
    fa : Any
        Number of false alarms (scalar or array).
    cn : Any
        Number of correct negatives (scalar or array).
    cost_loss_ratios : Any
        Array of cost/loss ratios.
    climatology : Any
        Climatological base rate (scalar or array).

    Returns
    -------
    Any
        Array or DataArray of REV values for each cost/loss ratio.
    """
    n = hits + misses + fa + cn
    alpha = np.asarray(cost_loss_ratios)
    s = (hits + misses) / n

    # Expand dimensions for broadcasting if inputs are arrays
    # If alpha is 1D and others are scalars, no problem.
    # If others are arrays (e.g. 2D spatial), we need to broadcast.
    if isinstance(hits, (np.ndarray, xr.DataArray)) and hits.ndim > 0:
        if isinstance(hits, xr.DataArray):
            # Use Xarray broadcasting: Create a DataArray for alpha
            # with a new dimension name 'cost_loss'
            alpha_da = xr.DataArray(
                alpha, coords={"cost_loss": alpha}, dims=["cost_loss"]
            )
            alpha_broadcast = alpha_da
        else:
            # Move alpha to a new dimension to broadcast over hits/misses (numpy)
            alpha_broadcast = alpha[(...,) + (np.newaxis,) * hits.ndim]
    else:
        alpha_broadcast = alpha

    # Expected Expense for Forecast
    e_fcst = alpha_broadcast * (hits + fa) / n + misses / n

    # Expected Expense for Climatology
    if isinstance(hits, (np.ndarray, xr.DataArray)):
        if isinstance(hits, xr.DataArray):
            e_clim = xr.where(alpha_broadcast < s, alpha_broadcast, s)
        else:
            e_clim = np.minimum(alpha_broadcast, s)
    else:
        e_clim = np.minimum(alpha, s)

    # Expected Expense for Perfect Forecast
    e_perf = alpha_broadcast * s

    # REV calculation
    numerator = e_clim - e_fcst
    denominator = e_clim - e_perf

    if isinstance(hits, (xr.DataArray, xr.Dataset)):
        rev = numerator / denominator
        rev = rev.where(denominator != 0, 0)
        return _update_history(rev, "Calculated REV")

    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator != 0,
    )
