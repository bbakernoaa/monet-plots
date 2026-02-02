import numpy as np
import xarray as xr
import dask.array as da
from typing import Tuple, Union, Dict, Any, Optional


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
    if any(isinstance(x, (xr.DataArray, xr.Dataset)) for x in [hits, misses]):
        res = hits / denominator
        res = res.where(denominator != 0, 0)
        return _update_history(res, "Calculated POD")

    return np.divide(
        hits,
        denominator,
        out=np.zeros_like(denominator, dtype=float)
        if isinstance(denominator, np.ndarray)
        else None,
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
    if any(isinstance(x, (xr.DataArray, xr.Dataset)) for x in [hits, fa]):
        res = fa / denominator
        res = res.where(denominator != 0, 0)
        return _update_history(res, "Calculated FAR")

    return np.divide(
        fa,
        denominator,
        out=np.zeros_like(denominator, dtype=float)
        if isinstance(denominator, np.ndarray)
        else None,
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
    if any(isinstance(x, (xr.DataArray, xr.Dataset)) for x in [hits, fa]):
        res = hits / denominator
        res = res.where(denominator != 0, 0)
        return _update_history(res, "Calculated Success Ratio")

    return np.divide(
        hits,
        denominator,
        out=np.zeros_like(denominator, dtype=float)
        if isinstance(denominator, np.ndarray)
        else None,
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
    if any(isinstance(x, (xr.DataArray, xr.Dataset)) for x in [hits, misses, fa]):
        res = hits / denominator
        res = res.where(denominator != 0, 0)
        return _update_history(res, "Calculated CSI")

    return np.divide(
        hits,
        denominator,
        out=np.zeros_like(denominator, dtype=float)
        if isinstance(denominator, np.ndarray)
        else None,
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
    if any(isinstance(x, (xr.DataArray, xr.Dataset)) for x in [hits, misses, fa]):
        res = numerator / denominator
        res = res.where(denominator != 0, 0)
        return _update_history(res, "Calculated Frequency Bias")

    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(denominator, dtype=float)
        if isinstance(denominator, np.ndarray)
        else None,
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
    if any(isinstance(x, (xr.DataArray, xr.Dataset)) for x in [fa, cn]):
        res = fa / denominator
        res = res.where(denominator != 0, 0)
        return _update_history(res, "Calculated POFD")

    return np.divide(
        fa,
        denominator,
        out=np.zeros_like(denominator, dtype=float)
        if isinstance(denominator, np.ndarray)
        else None,
        where=denominator != 0,
    )


def _auc_core(x: np.ndarray, y: np.ndarray) -> float:
    """Core logic for AUC calculation using trapezoidal rule.

    Parameters
    ----------
    x : np.ndarray
        x-coordinates.
    y : np.ndarray
        y-coordinates.

    Returns
    -------
    float
        The calculated AUC.
    """
    sort_idx = np.argsort(x)
    return np.trapezoid(y[sort_idx], x[sort_idx])


def compute_auc(
    x: Union[np.ndarray, xr.DataArray],
    y: Union[np.ndarray, xr.DataArray],
    dim: str = "threshold",
) -> Union[float, xr.DataArray]:
    """
    Calculates Area Under Curve (AUC) using the trapezoidal rule.

    Supports lazy evaluation via Dask when provided with xarray objects.

    Parameters
    ----------
    x : Union[np.ndarray, xr.DataArray]
        x-coordinates (e.g., POFD).
    y : Union[np.ndarray, xr.DataArray]
        y-coordinates (e.g., POD).
    dim : str, optional
        The dimension to integrate over, by default "threshold".

    Returns
    -------
    Union[float, xr.DataArray]
        The calculated AUC (float or xarray.DataArray).

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([0.0, 0.5, 1.0])
    >>> y = np.array([0.0, 0.8, 1.0])
    >>> compute_auc(x, y)
    0.65
    """
    if isinstance(x, (xr.DataArray, xr.Dataset)) or isinstance(
        y, (xr.DataArray, xr.Dataset)
    ):
        # Ensure we are working with DataArrays
        x_da = x if isinstance(x, xr.DataArray) else xr.DataArray(x)
        y_da = y if isinstance(y, xr.DataArray) else xr.DataArray(y)

        # Handle Xarray/Dask lazily using apply_ufunc
        # Core dimensions must be unchunked for parallelized ufunc
        res = xr.apply_ufunc(
            _auc_core,
            x_da.chunk({dim: -1}),
            y_da.chunk({dim: -1}),
            input_core_dims=[[dim], [dim]],
            dask="parallelized",
            output_dtypes=[float],
        )
        res.name = "auc"
        return _update_history(res, "Calculated AUC")

    return float(_auc_core(np.asarray(x), np.asarray(y)))


def compute_reliability_curve(
    forecasts: Union[np.ndarray, xr.DataArray],
    observations: Union[np.ndarray, xr.DataArray],
    n_bins: int = 10,
) -> Tuple[
    Union[np.ndarray, xr.DataArray],
    Union[np.ndarray, xr.DataArray],
    Union[np.ndarray, xr.DataArray],
]:
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
    forecasts: Union[np.ndarray, xr.DataArray],
    observations: Union[np.ndarray, xr.DataArray],
    n_bins: int = 10,
) -> Dict[str, Union[float, xr.DataArray]]:
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
        Number of bins for reliability curve, by default 10.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys 'reliability', 'resolution', 'uncertainty',
        and 'brier_score'.
    """
    # Use .size for dimensionality awareness
    if hasattr(forecasts, "size"):
        N = forecasts.size
    else:
        N = len(forecasts)

    base_rate = observations.mean()
    uncertainty = base_rate * (1.0 - base_rate)

    bin_centers, obs_freq, bin_counts = compute_reliability_curve(
        forecasts, observations, n_bins
    )

    # Filter out empty bins. Need to compute mask if it's Dask to allow indexing.
    # obs_freq is small (n_bins), so this is safe and necessary for Xarray.
    is_lazy = hasattr(obs_freq, "chunks") and obs_freq.chunks is not None
    if is_lazy:
        mask = (~np.isnan(obs_freq)).compute()
    else:
        mask = ~np.isnan(obs_freq)

    bin_centers = bin_centers[mask]
    obs_freq = obs_freq[mask]
    bin_counts = bin_counts[mask]

    # Reliability: Weighted average of (forecast - observed_freq)^2
    reliability = (bin_counts * (bin_centers - obs_freq) ** 2).sum() / N

    # Resolution: Weighted average of (observed_freq - base_rate)**2
    resolution = (bin_counts * (obs_freq - base_rate) ** 2).sum() / N

    bs = reliability - resolution + uncertainty

    res = {
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "brier_score": bs,
    }

    # Update history for all components if they are Xarray
    for key, value in res.items():
        if isinstance(value, (xr.DataArray, xr.Dataset)):
            _update_history(value, f"Computed Brier Score component: {key}")

    return res


def compute_rank_histogram(
    ensemble: Union[np.ndarray, xr.DataArray],
    observations: Union[np.ndarray, xr.DataArray],
) -> Union[np.ndarray, xr.DataArray]:
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
        ensemble = ensemble.data
    if isinstance(observations, xr.DataArray):
        observations = observations.data

    # ensemble < observations[:, np.newaxis] broadcast comparison
    # This works for both numpy and dask arrays
    obs_expanded = observations[:, np.newaxis]
    ranks = (ensemble < obs_expanded).sum(axis=1)

    if hasattr(ranks, "chunks"):
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
    hits: Union[float, np.ndarray, xr.DataArray],
    misses: Union[float, np.ndarray, xr.DataArray],
    fa: Union[float, np.ndarray, xr.DataArray],
    cn: Union[float, np.ndarray, xr.DataArray],
    cost_loss_ratios: Union[np.ndarray, xr.DataArray],
    climatology: float | None = None,
) -> Union[np.ndarray, xr.DataArray]:
    """
    Calculates Relative Economic Value (REV).

    REV = (E_clim - E_forecast) / (E_clim - E_perfect)
    Where E is expected expense per event.

    Parameters
    ----------
    hits : Any
        Number of hits (scalar, numpy array, or xarray.DataArray).
    misses : Any
        Number of misses.
    fa : Any
        Number of false alarms.
    cn : Any
        Number of correct negatives.
    cost_loss_ratios : Any
        Array-like of cost/loss ratios [0, 1].
    climatology : float, optional
        Climatological base rate (hits + misses) / n. If None, it is
        calculated from the input contingency table, by default None.

    Returns
    -------
    Any
        Calculated REV. Returns xarray.DataArray if inputs are xarray.
    """
    n = hits + misses + fa + cn

    if climatology is not None:
        s = climatology
    else:
        s = (hits + misses) / n

    # Handle alpha broadcasting for Xarray
    is_xarray = any(
        isinstance(x, (xr.DataArray, xr.Dataset))
        for x in [hits, misses, fa, cn, cost_loss_ratios]
    )

    if is_xarray:
        if not isinstance(cost_loss_ratios, (xr.DataArray, xr.Dataset)):
            alpha = xr.DataArray(
                cost_loss_ratios,
                coords={"cost_loss_ratio": cost_loss_ratios},
                dims=["cost_loss_ratio"],
            )
        else:
            alpha = cost_loss_ratios
    else:
        alpha = np.asarray(cost_loss_ratios)

    # Expected Expense for Forecast
    e_fcst = alpha * (hits + fa) / n + misses / n

    # Expected Expense for Climatology
    if is_xarray:
        e_clim = xr.where(alpha < s, alpha, s)
    else:
        e_clim = np.minimum(alpha, s)

    # Expected Expense for Perfect Forecast
    e_perf = alpha * s

    # REV calculation
    numerator = e_clim - e_fcst
    denominator = e_clim - e_perf

    if is_xarray:
        rev = numerator / denominator
        rev = rev.where(denominator != 0, 0)
        return _update_history(rev, "Calculated Relative Economic Value (REV)")

    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator != 0,
    )
