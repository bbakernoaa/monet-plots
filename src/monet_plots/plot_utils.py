import warnings
from typing import Any, Optional


import numpy as np
import pandas as pd

# Optional xarray import - will be used if available
try:
    import xarray as xr
except ImportError:
    xr = None


def to_dataframe(data: Any) -> pd.DataFrame:
    """
    Convert input data to a pandas DataFrame.

    Args:
        data: Input data. Can be a pandas DataFrame, xarray DataArray,
              xarray Dataset, or numpy ndarray.

    Returns:
        A pandas DataFrame.

    Raises:
        TypeError: If the input data type is not supported.
    """
    if isinstance(data, pd.DataFrame):
        return data

    # Using hasattr to avoid direct dependency on xarray for users who don't have it
    # installed.
    if hasattr(data, "to_dataframe"):  # Works for both xarray DataArray and Dataset
        return data.to_dataframe()

    if isinstance(data, np.ndarray):
        if data.ndim == 1:
            return pd.DataFrame(data, columns=["col_0"])
            return pd.DataFrame(data, columns=["col_0"])
        elif data.ndim == 2:
            return pd.DataFrame(
                data, columns=[f"col_{i}" for i in range(data.shape[1])]
            )
            return pd.DataFrame(
                data, columns=[f"col_{i}" for i in range(data.shape[1])]
            )
        else:
            raise ValueError(f"numpy array with {data.ndim} dimensions not supported")

    raise TypeError(f"Unsupported data type: {type(data).__name__}")


def _validate_spatial_plot_params(kwargs):
    """Validate parameters specific to SpatialPlot."""
    if "discrete" in kwargs:
        discrete = kwargs["discrete"]
        if not isinstance(discrete, bool):
            raise TypeError(
                f"discrete parameter must be boolean, got {type(discrete).__name__}"
            )

    if "ncolors" in kwargs:
        ncolors = kwargs["ncolors"]
        if not isinstance(ncolors, int):
            raise TypeError(
                f"ncolors parameter must be integer, got {type(ncolors).__name__}"
            )
        if ncolors <= 0 or ncolors > 1000:
            raise ValueError(
                f"ncolors parameter must be between 1 and 1000, got {ncolors}"
            )

    _validate_plotargs(kwargs.get("plotargs"))


def _validate_timeseries_plot_params(kwargs):
    """Validate parameters specific to TimeSeriesPlot."""
    if "x" in kwargs:
        x = kwargs["x"]
        if not isinstance(x, str):
            raise TypeError(f"x parameter must be string, got {type(x).__name__}")

    if "y" in kwargs:
        y = kwargs["y"]
        if not isinstance(y, str):
            raise TypeError(f"y parameter must be string, got {type(y).__name__}")

    _validate_plotargs(kwargs.get("plotargs"))
    _validate_fillargs(kwargs.get("fillargs"))


def _validate_plotargs(plotargs):
    """Validate plotargs parameter."""
    if plotargs is not None:
        if not isinstance(plotargs, dict):
            raise TypeError(
                f"plotargs parameter must be dict, got {type(plotargs).__name__}"
            )

        if "cmap" in plotargs:
            cmap = plotargs["cmap"]
            if not isinstance(cmap, str):
                raise TypeError(f"colormap must be string, got {type(cmap).__name__}")


def _validate_fillargs(fillargs):
    """Validate fillargs parameter."""
    if fillargs is not None:
        if not isinstance(fillargs, dict):
            raise TypeError(
                f"fillargs parameter must be dict, got {type(fillargs).__name__}"
            )

        if "alpha" in fillargs:
            alpha = fillargs["alpha"]
            if not isinstance(alpha, (int, float)):
                raise TypeError(f"alpha must be numeric, got {type(alpha).__name__}")
            if not 0 <= alpha <= 1:
                raise ValueError(f"alpha must be between 0 and 1, got {alpha}")


def _validate_spatial_plot_params(kwargs):
    """Validate parameters specific to SpatialPlot."""
    if "discrete" in kwargs:
        discrete = kwargs["discrete"]
        if not isinstance(discrete, bool):
            raise TypeError(
                f"discrete parameter must be boolean, got {type(discrete).__name__}"
            )

    if "ncolors" in kwargs:
        ncolors = kwargs["ncolors"]
        if not isinstance(ncolors, int):
            raise TypeError(
                f"ncolors parameter must be integer, got {type(ncolors).__name__}"
            )
        if ncolors <= 0 or ncolors > 1000:
            raise ValueError(
                f"ncolors parameter must be between 1 and 1000, got {ncolors}"
            )

    _validate_plotargs(kwargs.get("plotargs"))


def _validate_timeseries_plot_params(kwargs):
    """Validate parameters specific to TimeSeriesPlot."""
    if "x" in kwargs:
        x = kwargs["x"]
        if not isinstance(x, str):
            raise TypeError(f"x parameter must be string, got {type(x).__name__}")

    if "y" in kwargs:
        y = kwargs["y"]
        if not isinstance(y, str):
            raise TypeError(f"y parameter must be string, got {type(y).__name__}")

    _validate_plotargs(kwargs.get("plotargs"))
    _validate_fillargs(kwargs.get("fillargs"))


def _validate_plotargs(plotargs):
    """Validate plotargs parameter."""
    if plotargs is not None:
        if not isinstance(plotargs, dict):
            raise TypeError(
                f"plotargs parameter must be dict, got {type(plotargs).__name__}"
            )

        if "cmap" in plotargs:
            cmap = plotargs["cmap"]
            if not isinstance(cmap, str):
                raise TypeError(f"colormap must be string, got {type(cmap).__name__}")


def _validate_fillargs(fillargs):
    """Validate fillargs parameter."""
    if fillargs is not None:
        if not isinstance(fillargs, dict):
            raise TypeError(
                f"fillargs parameter must be dict, got {type(fillargs).__name__}"
            )

        if "alpha" in fillargs:
            alpha = fillargs["alpha"]
            if not isinstance(alpha, (int, float)):
                raise TypeError(f"alpha must be numeric, got {type(alpha).__name__}")
            if not 0 <= alpha <= 1:
                raise ValueError(f"alpha must be between 0 and 1, got {alpha}")


def validate_plot_parameters(plot_class: str, method: str, **kwargs) -> None:
    """
    Validate parameters for plot methods.

    Args:
        plot_class: The plot class name
        method: The method name
        **kwargs: Parameters to validate

    Raises:
        TypeError: If parameter types are invalid
        ValueError: If parameter values are invalid
    """
    if plot_class == "SpatialPlot" and method == "plot":
        _validate_spatial_plot_params(kwargs)
    elif plot_class == "TimeSeriesPlot" and method == "plot":
        _validate_timeseries_plot_params(kwargs)
    if plot_class == "SpatialPlot" and method == "plot":
        _validate_spatial_plot_params(kwargs)
    elif plot_class == "TimeSeriesPlot" and method == "plot":
        _validate_timeseries_plot_params(kwargs)


def validate_data_array(data: Any, required_dims: Optional[list] = None) -> None:
    """
    Validate data array parameters.

    Args:
        data: Data to validate
        required_dims: List of required dimension names

    Raises:
        TypeError: If data type is invalid
        ValueError: If data dimensions are invalid
    """
    if data is None:
        raise ValueError("data cannot be None")

    # Check if data has required attributes
    if not hasattr(data, "shape"):
        raise TypeError("data must have a shape attribute")

    if required_dims:
        if not hasattr(data, "dims"):
            raise TypeError("data must have dims attribute for dimension validation")

        for dim in required_dims:
            if dim not in data.dims:
                raise ValueError(
                    f"required dimension '{dim}' not found in data dimensions {data.dims}"
                )
                raise ValueError(
                    f"required dimension '{dim}' not found in data dimensions {data.dims}"
                )


def validate_dataframe(df: Any, required_columns: Optional[list] = None) -> None:
    """
    Validate DataFrame parameters.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names

    Raises:
        TypeError: If DataFrame type is invalid
        ValueError: If DataFrame structure is invalid
    """
    if df is None:
        raise ValueError("DataFrame cannot be None")

    if not hasattr(df, "columns"):
        raise TypeError("object must have columns attribute")

    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"missing required columns: {missing_columns}")

    if len(df) == 0:
        raise ValueError("DataFrame cannot be empty")


def _try_xarray_conversion(data):
    """Try to convert data to xarray format."""
    if xr is None:
        return None

    # Check if already xarray
    if hasattr(xr, "DataArray") and isinstance(data, xr.DataArray):
        return data
    if hasattr(xr, "Dataset") and isinstance(data, xr.Dataset):
        return data

    # Try xarray-like conversion
    if hasattr(data, "to_dataset") and hasattr(data, "to_dataframe"):
        try:
            return data.to_dataset()
        except Exception:
            return None

    return None


def _convert_numpy_to_dataframe(data):
    """Convert numpy array to DataFrame."""
    if data.ndim == 1:
        return pd.DataFrame(data, columns=["col_0"])
    elif data.ndim == 2:
        return pd.DataFrame(data, columns=[f"col_{i}" for i in range(data.shape[1])])
    else:
        raise ValueError(f"numpy array with {data.ndim} dimensions not supported")


def _try_xarray_conversion(data):
    """Try to convert data to xarray format."""
    if xr is None:
        return None

    # Check if already xarray
    if hasattr(xr, "DataArray") and isinstance(data, xr.DataArray):
        return data
    if hasattr(xr, "Dataset") and isinstance(data, xr.Dataset):
        return data

    # Try xarray-like conversion
    if hasattr(data, "to_dataset") and hasattr(data, "to_dataframe"):
        try:
            return data.to_dataset()
        except Exception:
            return None

    return None


def _convert_numpy_to_dataframe(data):
    """Convert numpy array to DataFrame."""
    if data.ndim == 1:
        return pd.DataFrame(data, columns=["col_0"])
    elif data.ndim == 2:
        return pd.DataFrame(data, columns=[f"col_{i}" for i in range(data.shape[1])])
    else:
        raise ValueError(f"numpy array with {data.ndim} dimensions not supported")


def _normalize_data(data: Any) -> Any:
    """
    Normalize input data to a standardized format, preferring xarray objects when possible.

    This function intelligently handles different input types:
    - xarray DataArray/Dataset: returned as-is (preferred format)
    - pandas DataFrame: returned as-is
    - numpy array: converted to DataFrame
    - Other types: converted to DataFrame if possible

    Args:
        data: Input data of various types

    Returns:
        Either an xarray DataArray, xarray Dataset, or pandas DataFrame

    Raises:
        TypeError: If the input data type is not supported
    """
    # Try xarray conversion first
    xarray_result = _try_xarray_conversion(data)
    if xarray_result is not None:
        return xarray_result
    # Try xarray conversion first
    xarray_result = _try_xarray_conversion(data)
    if xarray_result is not None:
        return xarray_result

    # Check if data is a pandas DataFrame
    if isinstance(data, pd.DataFrame):
        return data

    # Check if data is numpy array
    if isinstance(data, np.ndarray):
        return _convert_numpy_to_dataframe(data)
        return _convert_numpy_to_dataframe(data)

    # Fall back to existing to_dataframe logic for backward compatibility
    return to_dataframe(data)


def normalize_data(data: Any) -> Any:
    """
    Public API for normalizing data, preferring xarray objects when possible.

    This is the same as _normalize_data but exposed as a public API.

    Args:
        data: Input data of various types

    Returns:
        Either an xarray DataArray, xarray Dataset, or pandas DataFrame
    """
    return _normalize_data(data)


def get_plot_kwargs(cmap: Any = None, norm: Any = None, **kwargs: Any) -> dict:
    """
    Helper to prepare keyword arguments for plotting functions.

    This function handles cases where `cmap` might be a tuple of
    (colormap, norm) returned by the scaling tools in `colorbars.py`.

    Parameters
    ----------
    cmap : Any, optional
        Colormap name, object, or (colormap, norm) tuple.
    norm : Any, optional
        Normalization object.
    **kwargs : Any
        Additional keyword arguments.

    Returns
    -------
    dict
        A dictionary of keyword arguments suitable for matplotlib plotting functions.
    """
    if isinstance(cmap, tuple) and len(cmap) == 2:
        kwargs["cmap"] = cmap[0]
        kwargs["norm"] = cmap[1]
    elif cmap is not None:
        kwargs["cmap"] = cmap

    if norm is not None:
        kwargs["norm"] = norm

    return kwargs


def _dynamic_fig_size(obj):
    """Try to determine a generic figure size based on the shape of obj

    Parameters
    ----------
    obj : A 2D xarray DataArray
        Description of parameter `obj`.

    Returns
    -------
    type
        Description of returned object.

    """
    scale = 1.0  # Default scale

    if "x" in obj.dims:
        nx, ny = len(obj.x), len(obj.y)
        scale = float(ny) / float(nx)
    elif "latitude" in obj.dims:
        nx, ny = len(obj.longitude), len(obj.latitude)
        scale = float(ny) / float(nx)
    elif "lat" in obj.dims:
        nx, ny = len(obj.lon), len(obj.lat)
        scale = float(ny) / float(nx)

    figsize = (10, 10 * scale)
    return figsize


def _set_outline_patch_alpha(ax, alpha=0):
    """Set the transparency of map outline patches for Cartopy GeoAxes.

    This function attempts multiple methods to set the alpha (transparency) of
    map outlines when using Cartopy, handling different versions and configurations.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or cartopy.mpl.geoaxes.GeoAxes
        The axes object whose outline transparency should be modified.
    alpha : float, default 0
        Alpha value between 0 (fully transparent) and 1 (fully opaque).

    Notes
    -----
    The function tries multiple approaches to accommodate different Cartopy versions
    and configurations. If all attempts fail, a warning is issued.
    """
    for f in [
        lambda alpha: ax.axes.outline_patch.set_alpha(alpha),
        lambda alpha: ax.outline_patch.set_alpha(alpha),
        lambda alpha: ax.spines["geo"].set_alpha(alpha),
    ]:
        try:
            f(alpha)
        except AttributeError:
            continue
        else:
            break
    else:
        warnings.warn("unable to set outline_patch alpha", stacklevel=2)
