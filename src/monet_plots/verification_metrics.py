import numpy as np
import xarray as xr
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
    member_dim: str = "member",
) -> Union[np.ndarray, xr.DataArray]:
    """
    Computes rank histogram counts.

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
    if isinstance(ensemble, xr.DataArray) and isinstance(observations, xr.DataArray):
        # Use xarray's dimension-aware broadcasting
        ranks = (ensemble < observations).sum(dim=member_dim)
        n_members = ensemble.sizes[member_dim]

        # Flatten ranks for histogram (preserving dask if present)
        ranks_flat = ranks.data.ravel()

        if hasattr(ranks_flat, "chunks"):
            import dask.array as da

            counts, _ = da.histogram(ranks_flat, bins=np.arange(n_members + 2) - 0.5)
        else:
            counts = np.bincount(ranks_flat.astype(int), minlength=n_members + 1)

        counts_xr = xr.DataArray(
            counts,
            coords={"rank": np.arange(len(counts))},
            dims=["rank"],
            name="rank_counts",
        )
        return _update_history(counts_xr, "Computed rank histogram (dimension-aware)")

    # Fallback for numpy or mixed (including plain dask arrays)
    is_dask = hasattr(ensemble, "chunks") or hasattr(observations, "chunks")

    # Assume member is the last dimension for fallback
    # Or if 2D/1D pair, assume (n_samples, n_members) for backward compatibility
    if ensemble.ndim == 2 and observations.ndim == 1:
        obs_expanded = observations[:, np.newaxis]
        ranks = (ensemble < obs_expanded).sum(axis=1)
        n_members = ensemble.shape[1]
    else:
        # Generic case: assume last axis is members
        obs_expanded = np.expand_dims(observations, axis=-1)
        ranks = (ensemble < obs_expanded).sum(axis=-1)
        n_members = ensemble.shape[-1]

    if is_dask:
        import dask.array as da

        counts, _ = da.histogram(ranks.ravel(), bins=np.arange(n_members + 2) - 0.5)
    else:
        counts = np.bincount(ranks.astype(int).ravel(), minlength=n_members + 1)

    if isinstance(ensemble, (xr.DataArray, xr.Dataset)):
        counts_xr = xr.DataArray(
            counts,
            coords={"rank": np.arange(len(counts))},
            dims=["rank"],
            name="rank_counts",
        )
        return _update_history(counts_xr, "Computed rank histogram")

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
