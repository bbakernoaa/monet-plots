import warnings
from typing import Any, Optional

import numpy as np
import pandas as pd
import xarray as xr


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

    if hasattr(data, "to_dataframe"):  # Works for both xarray DataArray and Dataset
        return data.to_dataframe()

    if isinstance(data, np.ndarray):
        return _convert_numpy_to_dataframe(data)

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
    # Check if already xarray
    if isinstance(data, xr.DataArray):
        return data
    if isinstance(data, xr.Dataset):
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

    # Check if data is a pandas DataFrame
    if isinstance(data, pd.DataFrame):
        return data

    # Check if data is numpy array
    if isinstance(data, np.ndarray):
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


def identify_coords(data: xr.DataArray | xr.Dataset) -> tuple[str, str]:
    """Identify latitude and longitude coordinates in an xarray object.

    Uses CF conventions (units, axis, standard_name) and common naming
    patterns to find the spatial dimensions.

    Parameters
    ----------
    data : xr.DataArray or xr.Dataset
        The data object to inspect.

    Returns
    -------
    tuple[str, str]
        A tuple of (lat_name, lon_name).

    Raises
    ----------
    ValueError
        If coordinates cannot be unambiguously identified.
    """
    lat_names = ["lat", "latitude", "y", "lat_2d"]
    lon_names = ["lon", "longitude", "x", "lon_2d"]

    def is_lat(c):
        # Check units
        units = c.attrs.get("units", "").lower()
        if "degree" in units and ("north" in units or "n" == units):
            return True
        # Check standard_name
        if c.attrs.get("standard_name", "") == "latitude":
            return True
        # Check axis
        if c.attrs.get("axis", "") == "Y":
            return True
        # Check common names
        if any(name == str(c.name).lower() for name in lat_names):
            return True
        return False

    def is_lon(c):
        # Check units
        units = c.attrs.get("units", "").lower()
        if "degree" in units and ("east" in units or "e" == units):
            return True
        # Check standard_name
        if c.attrs.get("standard_name", "") == "longitude":
            return True
        # Check axis
        if c.attrs.get("axis", "") == "X":
            return True
        # Check common names
        if any(name == str(c.name).lower() for name in lon_names):
            return True
        return False

    found_lat = None
    found_lon = None

    # Check coordinates first
    for name in data.coords:
        coord = data.coords[name]
        if is_lat(coord):
            found_lat = name
        if is_lon(coord):
            found_lon = name

    # Fallback to dims if not found in coords (though usually they are the same)
    if not found_lat or not found_lon:
        for name in data.dims:
            if name in data.coords:
                continue  # already checked
            # If it's a dim but not a coord, it's harder to check attrs
            # but we can check the name
            if not found_lat and name.lower() in lat_names:
                found_lat = name
            if not found_lon and name.lower() in lon_names:
                found_lon = name

    if not found_lat or not found_lon:
        raise ValueError(
            f"Could not identify spatial coordinates. Found lat={found_lat}, lon={found_lon}. "
            f"Available coords: {list(data.coords.keys())}"
        )

    return found_lat, found_lon


def ensure_monotonic(data: xr.DataArray, lat_name: str, lon_name: str) -> xr.DataArray:
    """Ensure latitude is increasing and data is properly oriented.

    Parameters
    ----------
    data : xr.DataArray
        The data to orient.
    lat_name : str
        The name of the latitude coordinate.
    lon_name : str
        The name of the longitude coordinate.

    Returns
    -------
    xr.DataArray
        The oriented data array.
    """
    # We only handle 1D coordinates for monotonicity check for now
    # If coordinates are 2D, we assume they are correct or handled by the transform
    if data[lat_name].ndim == 1:
        if data[lat_name][0] > data[lat_name][-1]:
            data = data.sortby(lat_name, ascending=True)

    if data[lon_name].ndim == 1:
        # For longitude, we often want to ensure it's -180 to 180 or 0 to 360 consistently
        # But most importantly, it should be monotonic
        if data[lon_name][0] > data[lon_name][-1]:
            data = data.sortby(lon_name, ascending=True)

    return data


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
