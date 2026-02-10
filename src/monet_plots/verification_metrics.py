import numpy as np
import xarray as xr

try:
    import dask.array as da
except ImportError:
    da = None
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


def compute_mfb(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    dim: Optional[Union[str, list[str]]] = None,
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculates Mean Fractional Bias (MFB).

    MFB = 200 * Mean((mod - obs) / (mod + obs))

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
        The calculated MFB.
    """
    denominator = (mod + obs).astype(float)
    numerator = 200.0 * (mod - obs)

    if isinstance(obs, (xr.DataArray, xr.Dataset)):
        res = (numerator / denominator).mean(dim=dim)
        res = res.where(denominator.mean(dim=dim) != 0, 0)
        return _update_history(res, "Calculated Mean Fractional Bias")

    # Fallback for numpy/pandas. Converting to numpy avoids pandas axis issues with axis=()
    obs_vals = np.asarray(obs)
    mod_vals = np.asarray(mod)
    denominator_vals = (mod_vals + obs_vals).astype(float)
    numerator_vals = 200.0 * (mod_vals - obs_vals)

    return np.mean(
        np.divide(
            numerator_vals,
            denominator_vals,
            out=np.zeros_like(denominator_vals, dtype=float),
            where=denominator_vals != 0,
        ),
        axis=dim,
    )


def compute_mfe(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    dim: Optional[Union[str, list[str]]] = None,
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculates Mean Fractional Error (MFE).

    MFE = 200 * Mean(abs(mod - obs) / (mod + obs))

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
        The calculated MFE.
    """
    denominator = (mod + obs).astype(float)
    numerator = 200.0 * np.abs(mod - obs)

    if isinstance(obs, (xr.DataArray, xr.Dataset)):
        res = (numerator / denominator).mean(dim=dim)
        res = res.where(denominator.mean(dim=dim) != 0, 0)
        return _update_history(res, "Calculated Mean Fractional Error")

    # Fallback for numpy/pandas
    obs_vals = np.asarray(obs)
    mod_vals = np.asarray(mod)
    denominator_vals = (mod_vals + obs_vals).astype(float)
    numerator_vals = 200.0 * np.abs(mod_vals - obs_vals)

    return np.mean(
        np.divide(
            numerator_vals,
            denominator_vals,
            out=np.zeros_like(denominator_vals, dtype=float),
            where=denominator_vals != 0,
        ),
        axis=dim,
    )


def compute_nmb(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    dim: Optional[Union[str, list[str]]] = None,
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculates Normalized Mean Bias (NMB).

    NMB = 100 * Sum(mod - obs) / Sum(obs)

    Parameters
    ----------
    obs : Union[np.ndarray, xr.DataArray]
        Observed values.
    mod : Union[np.ndarray, xr.DataArray]
        Model values.
    dim : str or list of str, optional
        The dimension(s) over which to calculate the sum.

    Returns
    -------
    Union[float, np.ndarray, xr.DataArray]
        The calculated NMB.
    """
    if isinstance(obs, (xr.DataArray, xr.Dataset)):
        denominator = obs.sum(dim=dim).astype(float)
        numerator = 100.0 * (mod - obs).sum(dim=dim)
        res = numerator / denominator
        res = res.where(denominator != 0, 0)
        return _update_history(res, "Calculated Normalized Mean Bias")

    # Fallback for numpy/pandas
    obs_vals = np.asarray(obs)
    mod_vals = np.asarray(mod)
    denominator = np.sum(obs_vals, axis=dim).astype(float)
    numerator = 100.0 * np.sum(mod_vals - obs_vals, axis=dim)

    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator != 0,
    )


def compute_nme(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    dim: Optional[Union[str, list[str]]] = None,
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculates Normalized Mean Error (NME).

    NME = 100 * Sum(abs(mod - obs)) / Sum(obs)

    Parameters
    ----------
    obs : Union[np.ndarray, xr.DataArray]
        Observed values.
    mod : Union[np.ndarray, xr.DataArray]
        Model values.
    dim : str or list of str, optional
        The dimension(s) over which to calculate the sum.

    Returns
    -------
    Union[float, np.ndarray, xr.DataArray]
        The calculated NME.
    """
    if isinstance(obs, (xr.DataArray, xr.Dataset)):
        denominator = obs.sum(dim=dim).astype(float)
        numerator = 100.0 * np.abs(mod - obs).sum(dim=dim)
        res = numerator / denominator
        res = res.where(denominator != 0, 0)
        return _update_history(res, "Calculated Normalized Mean Error")

    # Fallback for numpy/pandas
    obs_vals = np.asarray(obs)
    mod_vals = np.asarray(mod)
    denominator = np.sum(obs_vals, axis=dim).astype(float)
    numerator = 100.0 * np.sum(np.abs(mod_vals - obs_vals), axis=dim)

    return np.divide(
        numerator,
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
    is_dask = (
        hasattr(forecasts, "chunks")
        or (isinstance(forecasts, xr.DataArray) and forecasts.chunks is not None)
    ) and da is not None

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


def compute_brier_skill_score(
    forecasts: Union[np.ndarray, xr.DataArray],
    observations: Union[np.ndarray, xr.DataArray],
    n_bins: int = 10,
    reference_bs: Optional[Union[float, np.ndarray, xr.DataArray]] = None,
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculates Brier Skill Score (BSS).

    BSS = 1 - (BS_forecast / BS_reference)
    By default, the reference is climatology: BS_ref = mean_obs * (1 - mean_obs).

    Parameters
    ----------
    forecasts : Union[np.ndarray, xr.DataArray]
        Array-like of forecast probabilities [0, 1].
    observations : Union[np.ndarray, xr.DataArray]
        Array-like of binary outcomes (0 or 1).
    n_bins : int, optional
        Number of bins for Brier Score decomposition, by default 10.
    reference_bs : Union[float, np.ndarray, xr.DataArray], optional
        Reference Brier Score. If None, it is calculated from climatology.

    Returns
    -------
    Union[float, np.ndarray, xr.DataArray]
        The calculated Brier Skill Score. Returns xarray.DataArray if inputs
        are xarray.

    Examples
    --------
    >>> import numpy as np
    >>> # Perfect forecast relative to bin centers
    >>> fcst = np.array([0.05, 0.95])
    >>> obs = np.array([0, 1])
    >>> compute_brier_skill_score(fcst, obs, n_bins=10)
    1.0
    """
    components = compute_brier_score_components(forecasts, observations, n_bins=n_bins)
    bs = components["brier_score"]

    if reference_bs is None:
        reference_bs = components["uncertainty"]

    if isinstance(bs, (xr.DataArray, xr.Dataset)):
        bss = 1.0 - (bs / reference_bs)
        bss = bss.where(reference_bs != 0, 0)
        return _update_history(bss, "Calculated Brier Skill Score")

    # Fallback for numpy or dask
    if (hasattr(bs, "chunks") or hasattr(reference_bs, "chunks")) and da is not None:
        ratio = da.where(reference_bs != 0, bs / reference_bs, 1.0)
        return 1.0 - ratio

    return 1.0 - np.divide(
        bs,
        reference_bs,
        out=np.ones_like(bs, dtype=float),
        where=reference_bs != 0,
    )


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

    # Calculate components using sum to maintain lazy evaluation.
    # We avoid explicit masking with .compute() by using fillna or nansum.

    # Reliability: Weighted average of (forecast - observed_freq)^2
    sq_diff_rel = (bin_centers - obs_freq) ** 2
    if isinstance(sq_diff_rel, xr.DataArray):
        reliability = (bin_counts * sq_diff_rel.fillna(0)).sum() / N
    else:
        reliability = np.nansum(bin_counts * sq_diff_rel) / N

    # Resolution: Weighted average of (observed_freq - base_rate)**2
    sq_diff_res = (obs_freq - base_rate) ** 2
    if isinstance(sq_diff_res, xr.DataArray):
        resolution = (bin_counts * sq_diff_res.fillna(0)).sum() / N
    else:
        resolution = np.nansum(bin_counts * sq_diff_res) / N

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


def compute_crp_skill_score(
    ensemble: Union[np.ndarray, xr.DataArray],
    observation: Union[np.ndarray, xr.DataArray],
    member_dim: str = "member",
    reference_crps: Optional[Union[float, np.ndarray, xr.DataArray]] = None,
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculates Continuous Ranked Probability Skill Score (CRPSS).

    CRPSS = 1 - (CRPS_forecast / CRPS_reference)
    If reference_crps is not provided, it must be provided by the user.

    Parameters
    ----------
    ensemble : Union[np.ndarray, xr.DataArray]
        Ensemble data. If xarray, it must have a dimension named `member_dim`.
    observation : Union[np.ndarray, xr.DataArray]
        Observation data.
    member_dim : str, optional
        The name of the ensemble member dimension, by default "member".
    reference_crps : Union[float, np.ndarray, xr.DataArray], optional
        Reference CRPS value. If None, a ValueError is raised.

    Returns
    -------
    Union[float, np.ndarray, xr.DataArray]
        The calculated CRPSS. Returns xarray.DataArray if inputs are xarray.

    Raises
    ------
    ValueError
        If reference_crps is not provided.
    """
    crps = compute_crps(ensemble, observation, member_dim=member_dim)

    if reference_crps is None:
        raise ValueError("reference_crps must be provided for CRPSS calculation.")

    if isinstance(crps, (xr.DataArray, xr.Dataset)):
        crpss = 1.0 - (crps / reference_crps)
        crpss = crpss.where(reference_crps != 0, 0)
        return _update_history(crpss, "Calculated CRPSS")

    # Fallback for numpy or dask
    if (
        hasattr(crps, "chunks") or hasattr(reference_crps, "chunks")
    ) and da is not None:
        ratio = da.where(reference_crps != 0, crps / reference_crps, 1.0)
        return 1.0 - ratio

    return 1.0 - np.divide(
        crps,
        reference_crps,
        out=np.ones_like(crps, dtype=float),
        where=reference_crps != 0,
    )


def compute_rps(
    forecast_probs: Union[np.ndarray, xr.DataArray],
    observed_probs: Union[np.ndarray, xr.DataArray],
    category_dim: str = "category",
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculates Ranked Probability Score (RPS).

    RPS = 1/(K-1) * Sum_{k=1}^{K-1} (CP_k - CO_k)^2
    where CP and CO are cumulative probabilities for forecast and observation.

    Parameters
    ----------
    forecast_probs : Union[np.ndarray, xr.DataArray]
        Forecast probabilities for each category. Sum over categories should be 1.
    observed_probs : Union[np.ndarray, xr.DataArray]
        Observed probabilities (usually one-hot encoded).
    category_dim : str, optional
        The name of the category dimension, by default "category".

    Returns
    -------
    Union[float, np.ndarray, xr.DataArray]
        The calculated RPS. Returns xarray.DataArray if inputs are xarray.

    Examples
    --------
    >>> import numpy as np
    >>> fcst = np.array([0.1, 0.7, 0.2])
    >>> obs = np.array([0, 1, 0])
    >>> compute_rps(fcst, obs)
    0.025
    """
    if isinstance(forecast_probs, xr.DataArray) and isinstance(observed_probs, xr.DataArray):
        # Cumulative sum along category dimension
        cp = forecast_probs.cumsum(dim=category_dim)
        co = observed_probs.cumsum(dim=category_dim)

        k = forecast_probs.sizes[category_dim]
        # We only sum up to K-1 categories
        sq_diff = (cp - co) ** 2
        # Use isel to exclude the last category if it's the 1.0/1.0 cumulative sum
        # but the standard formula often just sums all and the last is 0.
        # Actually, the standard formula is 1/(K-1) * sum_{k=1}^{K-1}.
        # If we sum all K categories, the last term is (1-1)^2 = 0,
        # so we can just sum all and divide by (K-1).
        rps = sq_diff.sum(dim=category_dim) / (k - 1)
        return _update_history(rps, f"Calculated RPS along {category_dim}")

    # Fallback for numpy
    cp = np.cumsum(forecast_probs, axis=-1)
    co = np.cumsum(observed_probs, axis=-1)
    k = forecast_probs.shape[-1]
    rps = np.sum((cp - co) ** 2, axis=-1) / (k - 1)

    if isinstance(forecast_probs, (xr.DataArray, xr.Dataset)):
        res = xr.DataArray(rps, name="rps")
        return _update_history(res, "Calculated RPS")

    return rps
