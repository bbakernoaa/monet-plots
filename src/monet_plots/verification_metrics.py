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


def compute_bias(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    dim: Optional[Union[str, list[str]]] = None,
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculates Mean Bias.

    Bias = Mean(mod - obs)

    Parameters
    ----------
    obs : Union[np.ndarray, xr.DataArray]
        Observed values.
    mod : Union[np.ndarray, xr.DataArray]
        Model values.
    dim : str or list of str, optional
        The dimension(s) over which to calculate the mean.

    Returns
    -------
    Union[float, np.ndarray, xr.DataArray]
        The calculated Mean Bias.

    Examples
    --------
    >>> import numpy as np
    >>> obs = np.array([1.0, 2.0, 3.0])
    >>> mod = np.array([1.1, 2.1, 3.1])
    >>> compute_bias(obs, mod)
    0.10000000000000009
    """
    diff = mod - obs
    if isinstance(diff, (xr.DataArray, xr.Dataset)):
        res = diff.mean(dim=dim)
        return _update_history(res, f"Calculated Mean Bias along {dim}")
    return np.mean(diff, axis=dim)


def compute_rmse(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    dim: Optional[Union[str, list[str]]] = None,
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculates Root Mean Square Error (RMSE).

    RMSE = sqrt(Mean((mod - obs)**2))

    Parameters
    ----------
    obs : Union[np.ndarray, xr.DataArray]
        Observed values.
    mod : Union[np.ndarray, xr.DataArray]
        Model values.
    dim : str or list of str, optional
        The dimension(s) over which to calculate the mean.

    Returns
    -------
    Union[float, np.ndarray, xr.DataArray]
        The calculated RMSE.

    Examples
    --------
    >>> import numpy as np
    >>> obs = np.array([1.0, 2.0])
    >>> mod = np.array([1.1, 2.2])
    >>> compute_rmse(obs, mod)
    0.158113883008419
    """
    diff_sq = (mod - obs) ** 2
    if isinstance(diff_sq, (xr.DataArray, xr.Dataset)):
        res = np.sqrt(diff_sq.mean(dim=dim))
        return _update_history(res, f"Calculated RMSE along {dim}")
    return np.sqrt(np.mean(diff_sq, axis=dim))


def compute_mae(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    dim: Optional[Union[str, list[str]]] = None,
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculates Mean Absolute Error (MAE).

    MAE = Mean(abs(mod - obs))

    Parameters
    ----------
    obs : Union[np.ndarray, xr.DataArray]
        Observed values.
    mod : Union[np.ndarray, xr.DataArray]
        Model values.
    dim : str or list of str, optional
        The dimension(s) over which to calculate the mean.

    Returns
    -------
    Union[float, np.ndarray, xr.DataArray]
        The calculated MAE.

    Examples
    --------
    >>> import numpy as np
    >>> obs = np.array([1.0, 2.0])
    >>> mod = np.array([1.1, 1.9])
    >>> compute_mae(obs, mod)
    0.10000000000000009
    """
    abs_diff = np.abs(mod - obs)
    if isinstance(abs_diff, (xr.DataArray, xr.Dataset)):
        res = abs_diff.mean(dim=dim)
        return _update_history(res, f"Calculated MAE along {dim}")
    return np.mean(abs_diff, axis=dim)


def compute_corr(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    dim: Optional[str] = None,
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculates Pearson correlation coefficient.

    Parameters
    ----------
    obs : Union[np.ndarray, xr.DataArray]
        Observed values.
    mod : Union[np.ndarray, xr.DataArray]
        Model values.
    dim : str, optional
        The dimension over which to calculate the correlation.

    Returns
    -------
    Union[float, np.ndarray, xr.DataArray]
        The calculated correlation.

    Examples
    --------
    >>> import numpy as np
    >>> obs = np.array([1, 2, 3])
    >>> mod = np.array([2, 4, 6])
    >>> compute_corr(obs, mod)
    1.0
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        res = xr.corr(obs, mod, dim=dim)
        return _update_history(res, f"Calculated correlation along {dim}")

    # Fallback to numpy
    if dim is not None:
        # np.corrcoef doesn't support dim directly like xarray
        # This is a simplification for 1D-like inputs or flattened
        return np.corrcoef(np.asarray(obs).ravel(), np.asarray(mod).ravel())[0, 1]

    return np.corrcoef(np.asarray(obs), np.asarray(mod))[0, 1]


def compute_auc(
    x: Union[np.ndarray, xr.DataArray],
    y: Union[np.ndarray, xr.DataArray],
    dim: Optional[str] = None,
) -> Union[float, xr.DataArray]:
    """
    Calculates Area Under Curve (AUC) using the trapezoidal rule.

    Supports lazy evaluation via Dask and multidimensional xarray objects.

    Parameters
    ----------
    x : Union[np.ndarray, xr.DataArray]
        x-coordinates (e.g., POFD).
    y : Union[np.ndarray, xr.DataArray]
        y-coordinates (e.g., POD).
    dim : str, optional
        The dimension along which to integrate. Required if x, y are
        multidimensional xarray objects. If not provided and inputs are
        1D xarray objects, the only dimension is used.

    Returns
    -------
    Union[float, xr.DataArray]
        The calculated AUC. Returns xarray.DataArray if inputs are xarray.
    """
    if isinstance(x, xr.DataArray) and isinstance(y, xr.DataArray):
        if dim is None:
            if x.ndim == 1:
                dim = x.dims[0]
            else:
                raise ValueError(
                    "dim must be provided for multidimensional xarray inputs"
                )

        # Ensure the integration dimension is not chunked for the ufunc
        x = x.chunk({dim: -1})
        y = y.chunk({dim: -1})

        def _auc_ufunc(x_arr, y_arr):
            # x_arr and y_arr have the core dimension as the last axis
            sort_idx = np.argsort(x_arr, axis=-1)
            x_sorted = np.take_along_axis(x_arr, sort_idx, axis=-1)
            y_sorted = np.take_along_axis(y_arr, sort_idx, axis=-1)
            return np.trapezoid(y_sorted, x_sorted, axis=-1)

        res = xr.apply_ufunc(
            _auc_ufunc,
            x,
            y,
            input_core_dims=[[dim], [dim]],
            dask="parallelized",
            output_dtypes=[float],
        )
        return _update_history(res, f"Calculated AUC along dimension {dim}")

    # Fallback for numpy or mixed
    x_val = np.asarray(x)
    y_val = np.asarray(y)

    # Ensure sorted by x
    sort_idx = np.argsort(x_val)
    auc = np.trapezoid(y_val[sort_idx], x_val[sort_idx])

    if isinstance(x, (xr.DataArray, xr.Dataset)) or isinstance(
        y, (xr.DataArray, xr.Dataset)
    ):
        res = xr.DataArray(auc, name="auc")
        return _update_history(res, "Calculated AUC")

    return float(auc)


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

    # Update history for all components if they are Xarray
    for key, value in res.items():
        if isinstance(value, (xr.DataArray, xr.Dataset)):
            _update_history(value, f"Computed Brier Score component: {key}")

    return res


def compute_rank_histogram(ensemble: Any, observations: Any) -> np.ndarray:
    """
    Computes rank histogram counts using vectorized operations.

    Supports multidimensional xarray inputs with automatic broadcasting.

    Parameters
    ----------
    ensemble : Union[np.ndarray, xr.DataArray]
        Ensemble data. If xarray, it must have a dimension named `member_dim`.
    observations : Union[np.ndarray, xr.DataArray]
        Observation data.
    member_dim : str, optional
        The name of the ensemble member dimension, by default "member".

    Returns
    -------
    Union[np.ndarray, xr.DataArray]
        Array or DataArray of counts for each rank (length n_members + 1).

    Examples
    --------
    >>> import numpy as np
    >>> ens = np.array([[1, 5], [2, 4], [3, 3]])
    >>> obs = np.array([2, 3, 4])
    >>> compute_rank_histogram(ens, obs)
    array([0, 2, 1])
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


def compute_crps(
    ensemble: Union[np.ndarray, xr.DataArray],
    observation: Union[np.ndarray, xr.DataArray],
    member_dim: str = "member",
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculates Continuous Ranked Probability Score (CRPS).

    CRPS measures the difference between the cumulative distribution function (CDF)
    of a probabilistic forecast and the empirical CDF of the observation.
    This implementation uses the efficient O(M log M) sorted ensemble method.

    Parameters
    ----------
    ensemble : Union[np.ndarray, xr.DataArray]
        Ensemble data. If xarray, it must have a dimension named `member_dim`.
    observation : Union[np.ndarray, xr.DataArray]
        Observation data.
    member_dim : str, optional
        The name of the ensemble member dimension, by default "member".

    Returns
    -------
    Union[float, np.ndarray, xr.DataArray]
        The calculated CRPS. Returns xarray.DataArray if inputs are xarray.

    Examples
    --------
    >>> import numpy as np
    >>> ens = np.array([1.0, 2.0, 3.0])
    >>> obs = 2.0
    >>> compute_crps(ens, obs)
    0.2222222222222222
    """

    def _crps_ufunc(ens_arr, obs_arr):
        # ens_arr has member dimension as last axis
        # obs_arr is scalar relative to the member dimension
        m = ens_arr.shape[-1]

        # Absolute difference from observation
        mae = np.mean(np.abs(ens_arr - np.expand_dims(obs_arr, axis=-1)), axis=-1)

        # Internal ensemble spread (Gini mean difference)
        ens_sorted = np.sort(ens_arr, axis=-1)
        i = np.arange(1, m + 1)
        # Using the formula: 2 * sum((2i - m - 1) * X_i) / m^2
        # Note: i is 1-based index
        spread = np.sum((2 * i - m - 1) * ens_sorted, axis=-1) / (m * m)

        return mae - spread

    if isinstance(ensemble, xr.DataArray) and isinstance(observation, xr.DataArray):
        # Ensure member dim is not chunked for the ufunc
        ensemble = ensemble.chunk({member_dim: -1})

        res = xr.apply_ufunc(
            _crps_ufunc,
            ensemble,
            observation,
            input_core_dims=[[member_dim], []],
            dask="parallelized",
            output_dtypes=[float],
        )
        return _update_history(res, f"Calculated CRPS (member_dim={member_dim})")

    # Fallback for numpy or mixed
    ens_val = np.asarray(ensemble)
    obs_val = np.asarray(observation)

    # Simple case for 1D ensemble and scalar observation
    if ens_val.ndim == 1 and obs_val.ndim == 0:
        return float(_crps_ufunc(ens_val, obs_val))

    # For more complex numpy shapes, we might need more logic,
    # but the ufunc logic is generally applicable if axes are aligned.
    # For simplicity, we assume the last axis is members if not xarray.
    res_val = _crps_ufunc(ens_val, obs_val)

    if isinstance(ensemble, (xr.DataArray, xr.Dataset)) or isinstance(
        observation, (xr.DataArray, xr.Dataset)
    ):
        res_xr = xr.DataArray(res_val, name="crps")
        return _update_history(res_xr, "Calculated CRPS")

    return res_val
