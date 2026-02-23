from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import monet_stats
import numpy as np
import xarray as xr


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
    Calculates Mean Bias using monet-stats.

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
    0.1
    """
    # monet_stats.MB calculates (obs - mod), so we swap mod/obs
    res = monet_stats.MB(mod, obs, axis=dim)
    if isinstance(res, (xr.DataArray, xr.Dataset)):
        return _update_history(res, f"Calculated Mean Bias along {dim}")
    return res


def compute_binned_bias(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    n_bins: int = 10,
    bin_range: Optional[tuple[float, float]] = None,
    dim: Optional[Union[str, list[str]]] = None,
) -> xr.Dataset:
    """
    Calculates mean bias binned by observed values.

    Parameters
    ----------
    obs : Union[np.ndarray, xr.DataArray]
        Observed values.
    mod : Union[np.ndarray, xr.DataArray]
        Model values.
    n_bins : int, optional
        Number of bins for observed values, by default 10.
    bin_range : tuple[float, float], optional
        The (min, max) range for the bins. If not provided and data is lazy,
        min/max will be computed from the data (triggering a compute).
        Providing `bin_range` ensures full laziness for Dask-backed inputs.
    dim : str or list of str, optional
        The dimension(s) over which to calculate the statistics.
        If None, all dimensions are used.

    Returns
    -------
    xr.Dataset
        Dataset containing 'bias_mean', 'bias_std', and 'count' binned by obs.
        The bins are represented by their centers in the 'bin_center' coordinate.
    """
    if not isinstance(obs, xr.DataArray):
        obs = xr.DataArray(obs)
    if not isinstance(mod, xr.DataArray):
        mod = xr.DataArray(mod)

    bias = mod - obs
    bias.name = "bias"

    # Determine bins. For dask-backed arrays, xarray requires explicit bin edges.
    # We also explicitly compute them for eager arrays to ensure parity.
    midpoints = None
    if isinstance(n_bins, int):
        if bin_range is not None:
            obs_min, obs_max = bin_range
        else:
            if hasattr(obs.data, "chunks"):
                import dask

                # We must compute min/max to establish the bin edges for the Dask graph.
                # Documented compute for lazy data when no bin_range is provided.
                obs_min, obs_max = dask.compute(obs.min(), obs.max())
            else:
                obs_min, obs_max = obs.min(), obs.max()

        bins = np.linspace(float(obs_min), float(obs_max), n_bins + 1)
        midpoints = (bins[:-1] + bins[1:]) / 2
    else:
        bins = n_bins
        if isinstance(bins, (np.ndarray, list)):
            bins_arr = np.asarray(bins)
            midpoints = (bins_arr[:-1] + bins_arr[1:]) / 2

    # Use xarray's groupby_bins, which leverages flox if installed for lazy dask support.
    binned = bias.groupby_bins(obs, bins=bins)

    res = xr.Dataset(
        {
            "bias_mean": binned.mean(dim=dim),
            "bias_std": binned.std(dim=dim),
            "bias_count": binned.count(dim=dim),
        }
    )

    # Convert Interval index to bin centers for plotting.
    bin_coords = [c for c in res.coords if "_bins" in str(c)]
    if bin_coords:
        bin_coord = bin_coords[0]
        if midpoints is not None:
            res = res.assign_coords({bin_coord: midpoints})
        else:
            # Fallback for complex bin specifications: metadata-only compute.
            midpoints = [i.mid for i in res.coords[bin_coord].values]
            res = res.assign_coords({bin_coord: midpoints})
        res = res.rename({bin_coord: "bin_center"})

    return _update_history(res, "Calculated binned bias")


def compute_binned_quantiles(
    obs: np.ndarray | xr.DataArray,
    mod: np.ndarray | xr.DataArray,
    quantiles: list[float] = [0.25, 0.5, 0.75],
    n_bins: int = 10,
    bin_range: tuple[float, float] | None = None,
    dim: str | list[str] | None = None,
) -> xr.Dataset:
    """
    Calculates quantiles of modeled values binned by observed values.

    Parameters
    ----------
    obs : np.ndarray | xr.DataArray
        Observed values.
    mod : np.ndarray | xr.DataArray
        Model values.
    quantiles : list[float], optional
        List of quantiles to calculate (0 to 1), by default [0.25, 0.5, 0.75].
    n_bins : int, optional
        Number of bins for observed values, by default 10.
    bin_range : tuple[float, float] | None, optional
        The (min, max) range for the bins. If not provided and data is lazy,
        min/max will be computed from the data (triggering a compute).
        Providing `bin_range` ensures full laziness for Dask-backed inputs.
    dim : str | list[str] | None, optional
        The dimension(s) over which to calculate the statistics.
        If None, all dimensions are used.

    Returns
    -------
    xr.Dataset
        Dataset containing binned quantiles. The quantiles are represented
        as a dimension, and bins are represented by their centers in the
        'bin_center' coordinate.

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> obs = np.linspace(0, 10, 100)
    >>> mod = obs + np.random.normal(0, 1, 100)
    >>> ds = compute_binned_quantiles(obs, mod, bins=5)
    """
    if not isinstance(obs, xr.DataArray):
        obs = xr.DataArray(obs)
    if not isinstance(mod, xr.DataArray):
        mod = xr.DataArray(mod)

    # Determine bins.
    midpoints = None
    if isinstance(n_bins, int):
        if bin_range is not None:
            obs_min, obs_max = bin_range
        else:
            if hasattr(obs.data, "chunks"):
                import dask

                obs_min, obs_max = dask.compute(obs.min(), obs.max())
            else:
                obs_min, obs_max = obs.min(), obs.max()

        bins = np.linspace(float(obs_min), float(obs_max), n_bins + 1)
        midpoints = (bins[:-1] + bins[1:]) / 2
    else:
        bins = n_bins
        if isinstance(bins, (np.ndarray, list)):
            bins_arr = np.asarray(bins)
            midpoints = (bins_arr[:-1] + bins_arr[1:]) / 2

    # Use xarray's groupby_bins for eager data, or where().quantile() for lazy data
    if hasattr(mod.data, "chunks") or hasattr(obs.data, "chunks"):
        # For Dask, flox + groupby + quantile is problematic.
        # We use a manual binning approach with where() to stay lazy.
        qs = []
        for i in range(len(bins) - 1):
            left, right = bins[i], bins[i + 1]
            mask = (obs > left) & (obs <= right)
            q = mod.where(mask).quantile(quantiles, dim=dim)
            qs.append(q)

        res = xr.concat(qs, dim="bin_center")
        res = res.assign_coords({"bin_center": midpoints})
    else:
        # Eager path remains same (uses flox/groupby_bins if available)
        binned = mod.groupby_bins(obs, bins=bins)
        res = binned.quantile(quantiles, dim=dim)

        # Convert Interval index to bin centers
        bin_coords = [c for c in res.coords if "_bins" in str(c)]
        if bin_coords:
            bin_coord = bin_coords[0]
            if midpoints is not None:
                res = res.assign_coords({bin_coord: midpoints})
            else:
                midpoints = [i.mid for i in res.coords[bin_coord].values]
                res = res.assign_coords({bin_coord: midpoints})
            res = res.rename({bin_coord: "bin_center"})

    # Convert to Dataset if it returned a DataArray (it usually does for quantile)
    if isinstance(res, xr.DataArray):
        res = res.to_dataset(name="mod_quantile")

    return _update_history(res, "Calculated binned quantiles")


def compute_rmse(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    dim: Optional[Union[str, list[str]]] = None,
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculates Root Mean Square Error (RMSE) using monet-stats.

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
    res = monet_stats.RMSE(obs, mod, axis=dim)
    if isinstance(res, (xr.DataArray, xr.Dataset)):
        return _update_history(res, f"Calculated RMSE along {dim}")
    return res


def compute_mae(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    dim: Optional[Union[str, list[str]]] = None,
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculates Mean Absolute Error (MAE) using monet-stats.

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
    0.1
    """
    res = monet_stats.MAE(obs, mod, axis=dim)
    if isinstance(res, (xr.DataArray, xr.Dataset)):
        return _update_history(res, f"Calculated MAE along {dim}")
    return res


def compute_mfb(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    dim: Optional[Union[str, list[str]]] = None,
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculates Mean Fractional Bias (MFB) using monet-stats.

    MFB = Mean(200 * (mod - obs) / (mod + obs))

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
    # For element-wise (dim=[]), monet_stats.FB uses axis=None which reduces.
    # We need to handle dim=[] explicitly if monet-stats doesn't support it for per-element.
    if dim == [] or dim == ():
        num = 200.0 * (mod - obs)
        den = mod + obs
        if isinstance(num, (xr.DataArray, xr.Dataset)):
            return (num / den).where(den != 0, np.nan)
        return np.divide(
            num, den, out=np.full(num.shape, np.nan, dtype=float), where=den != 0
        )

    res = monet_stats.FB(obs, mod, axis=dim)
    if isinstance(res, (xr.DataArray, xr.Dataset)):
        return _update_history(res, f"Calculated MFB along {dim}")
    return res


def compute_mfe(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    dim: Optional[Union[str, list[str]]] = None,
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculates Mean Fractional Error (MFE) using monet-stats.

    MFE = Mean(200 * abs(mod - obs) / (mod + obs))

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
    if dim == [] or dim == ():
        num = 200.0 * np.abs(mod - obs)
        den = mod + obs
        if isinstance(num, (xr.DataArray, xr.Dataset)):
            return (num / den).where(den != 0, np.nan)
        return np.divide(
            num, den, out=np.full(num.shape, np.nan, dtype=float), where=den != 0
        )

    res = monet_stats.FE(obs, mod, axis=dim)
    if isinstance(res, (xr.DataArray, xr.Dataset)):
        return _update_history(res, f"Calculated MFE along {dim}")
    return res


def compute_nmb(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    dim: Optional[Union[str, list[str]]] = None,
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculates Normalized Mean Bias (NMB) using monet-stats.

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
    if dim == [] or dim == ():
        num = 100.0 * (mod - obs)
        den = obs
        if isinstance(num, (xr.DataArray, xr.Dataset)):
            return (num / den).where(den != 0, np.nan)
        return np.divide(
            num, den, out=np.full(num.shape, np.nan, dtype=float), where=den != 0
        )

    res = monet_stats.NMB(obs, mod, axis=dim)
    if isinstance(res, (xr.DataArray, xr.Dataset)):
        return _update_history(res, f"Calculated NMB along {dim}")
    return res


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
    diff_abs = np.abs(mod - obs)
    if isinstance(diff_abs, (xr.DataArray, xr.Dataset)):
        num = diff_abs.sum(dim=dim)
        den = obs.sum(dim=dim)
        res = (100.0 * num / den).where(den != 0, np.nan)
        return _update_history(res, f"Calculated NME along {dim}")
    # Fallback for numpy/pandas
    if dim == [] or dim == ():
        num = diff_abs
        den = obs
    else:
        num = np.sum(diff_abs, axis=dim)
        den = np.sum(obs, axis=dim)
    return 100.0 * np.divide(
        num, den, out=np.full(np.shape(num), np.nan, dtype=float), where=den != 0
    )


def compute_corr(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    dim: Optional[str] = None,
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculates Pearson correlation coefficient using monet-stats.

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
    res = monet_stats.correlation(obs, mod, axis=dim)
    if isinstance(res, (xr.DataArray, xr.Dataset)):
        return _update_history(res, f"Calculated correlation along {dim}")
    return res


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
