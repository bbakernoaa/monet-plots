"""Coordinate and grid utilities shared by all spatial renderers.

These functions never create figures or compute the plotted data values.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def identify_coords(data, x=None, y=None):
    """Resolve spatial coordinates using overrides, CF metadata, then exact names.

    Projected X/Y coordinates are allowed; callers must supply their data CRS.
    Ambiguous coordinates require explicit names rather than an arbitrary choice.
    """
    items = dict(getattr(data, "coords", {}))
    items.update(getattr(data, "data_vars", {}))
    if hasattr(data, "columns"):
        items = {name: data[name] for name in data.columns}

    def resolve(explicit, axis):
        if explicit is not None:
            if explicit not in items:
                raise ValueError(f"Coordinate {explicit!r} not found in data.")
            return explicit
        standard = {
            "X": {"longitude", "projection_x_coordinate"},
            "Y": {"latitude", "projection_y_coordinate"},
        }[axis]
        units = {
            "X": {"degrees_east", "degree_east", "degree_e", "degrees_e"},
            "Y": {"degrees_north", "degree_north", "degree_n", "degrees_n"},
        }[axis]
        names = {"X": ("lon", "longitude", "x"), "Y": ("lat", "latitude", "y")}[axis]
        cf = [
            name
            for name, obj in items.items()
            if obj.attrs.get("standard_name", "").lower() in standard
            or obj.attrs.get("units", "").lower() in units
        ]
        candidates = cf or [
            name
            for name, obj in items.items()
            if obj.attrs.get("axis", "").upper() == axis
        ]
        if not candidates:
            candidates = [name for name in items if str(name).lower() in names[:2]]
        if not candidates:
            candidates = [name for name in items if str(name).lower() == names[2]]
        if len(candidates) != 1:
            raise ValueError(
                f"Could not identify a unique {'longitude' if axis == 'X' else 'latitude'} "
                f"coordinate (candidates: {candidates}). Supply x/y or lon_coord/lat_coord."
            )
        return candidates[0]

    return resolve(x, "X"), resolve(y, "Y")


def ordered_grid(data, x, y):
    """Sort only independent 1D grid dimensions; preserve station/curved grids."""
    if not isinstance(data, (xr.DataArray, xr.Dataset)):
        return data
    xc, yc = data[x], data[y]
    if xc.ndim == yc.ndim == 1 and xc.dims != yc.dims:
        lon = data[x]
        is_longitude = (
            str(x).lower() in {"lon", "longitude"}
            or lon.attrs.get("standard_name") == "longitude"
        )
        if is_longitude and np.any(np.abs(np.diff(lon.values)) > 180):
            unwrapped = np.rad2deg(np.unwrap(np.deg2rad(lon.values)))
            unwrapped -= 360 * np.floor((np.mean(unwrapped) + 180) / 360)
            data = data.assign_coords({x: (lon.dims, unwrapped, lon.attrs)})
        for name in (x, y):
            coord = data[name]
            if coord.size > 1 and np.any(np.diff(coord.values) < 0):
                data = data.sortby(name)
    return data


def grid_data(data, x, y):
    """Validate and transpose a field into y/x order, preserving lazy values."""
    if not isinstance(data, xr.DataArray) or data.ndim != 2:
        raise ValueError(
            "A spatial field must be a 2D DataArray; select or facet extra dimensions."
        )
    xc, yc = data[x], data[y]
    if xc.ndim == yc.ndim == 1 and xc.dims != yc.dims:
        return data.transpose(yc.dims[0], xc.dims[0])
    if xc.ndim == yc.ndim == 2 and set(xc.dims) == set(data.dims) == set(yc.dims):
        return data.transpose(*yc.dims)
    raise ValueError(
        "Spatial coordinates must be independent 1D axes or matching 2D grids."
    )


def raster_extent(data, x, y):
    """Get pixel-edge bounds for a regular grid, rejecting distorted rasters."""
    bounds = []
    for name in (x, y):
        coord = data[name]
        if coord.ndim != 1 or coord.size < 2:
            raise ValueError(
                "imshow requires regular 1D coordinates with at least two cells; use pcolormesh."
            )
        original = np.asarray(coord)
        values = original.astype(float)
        step = np.diff(values)
        # Coordinate quantization grows with magnitude, especially for float32.
        # Compare spacing at the precision actually stored, not float64 precision.
        eps = (
            np.finfo(original.dtype).eps
            if np.issubdtype(original.dtype, np.floating)
            else 0
        )
        tolerance = 4 * eps * max(1, float(np.max(np.abs(values))))
        pitch = (values[-1] - values[0]) / (len(values) - 1)
        if (
            not np.isfinite(values).all()
            or not np.allclose(step, pitch, rtol=1e-5, atol=tolerance)
            or np.any(step <= 0)
        ):
            raise ValueError(
                "imshow requires increasing, regularly spaced coordinates; use pcolormesh."
            )
        bounds.extend([values[0] - pitch / 2, values[-1] + pitch / 2])
    return bounds


def variable_label(data):
    """Build a colorbar label without discarding physical units."""
    name = data.attrs.get("long_name", data.name or "")
    units = data.attrs.get("units", "")
    return f"{name} ({units})" if units else str(name)


def validate_data_crs(data, x, y, data_crs):
    """Require an explicit CRS for coordinates identified as projected by CF."""
    if data_crs is not None:
        return
    for name in (x, y):
        coordinate = data[name]
        if coordinate.attrs.get("standard_name", "").startswith(
            "projection_"
        ) or coordinate.attrs.get("units", "").lower() in {
            "m",
            "metre",
            "meter",
            "metres",
            "meters",
            "km",
            "kilometres",
            "kilometers",
        }:
            raise ValueError(
                "Projected coordinates require an explicit data_crs; names and axis metadata do not specify a projection."
            )
