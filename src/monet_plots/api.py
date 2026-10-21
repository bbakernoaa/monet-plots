"""Convenience API for single maps and spatial facet comparisons."""

from __future__ import annotations

import cartopy.crs as ccrs
import pandas as pd
import xarray as xr

from .geo import identify_coords
from .plots.facet_grid import SpatialFacetGridPlot
from .plots.spatial import SpatialTrack
from .plots.spatial_contour import SpatialContourPlot
from .plots.spatial_imshow import SpatialImshowPlot
from .plots.spatial_mesh import SpatialMeshPlot


def spatial(
    data,
    *,
    kind="pcolormesh",
    row=None,
    col=None,
    col_wrap=None,
    var_name=None,
    x=None,
    y=None,
    colorbar="shared",
    observations=None,
    obs_var=None,
    obs_crs=None,
    obs_kwargs=None,
    panel_labels=False,
    **kwargs,
):
    """Render a map or facet comparison and return its customizable plot object.

    Parameters
    ----------
    data : xarray.DataArray, xarray.Dataset or pandas.DataFrame
        Gridded fields or, with kind='scatter', station/track samples.
    kind : {'pcolormesh', 'contourf', 'imshow', 'scatter'}
        Meshes support regular, irregular and curvilinear grids. Images require
        regular 1D coordinates. Extra field dimensions must be selected/faceted.
    row, col : str, optional
        Facet dimensions (xarray) or columns (pandas).
    var_name : str, optional
        Field to select from a Dataset or DataFrame.
    x, y : str, optional
        Explicit coordinate names; otherwise CF metadata and exact names apply.
    colorbar : {'shared', 'each', False}
        Facets share one scale unless 'each' is requested.
    observations : xarray.DataArray, xarray.Dataset or pandas.DataFrame, optional
        Points overlaid using each field's color scale. Shared facet coordinates
        are selected automatically; otherwise points are repeated in each panel.
        Observation values must have compatible units; no conversion is implied.
    obs_crs : cartopy.crs.CRS, optional
        Observation CRS, geographic PlateCarree by default, independent of the
        model's data_crs and the map's display projection.
    panel_labels : bool
        Add (a), (b), ... panel labels.
    **kwargs
        Map and rendering options, including projection, data_crs, extent_crs,
        extent, norm, cmap, vmin/vmax, rasterized, size/aspect and cbar_kwargs.

    Returns
    -------
    BasePlot
        Object exposing fig, axes, artists, colorbars, save() and close().
        Existing spatial class APIs remain available.
    """
    if "add_colorbar" in kwargs:
        colorbar = "shared" if kwargs.pop("add_colorbar") else False
    renderers = {
        "pcolormesh": SpatialMeshPlot,
        "contourf": SpatialContourPlot,
        "imshow": SpatialImshowPlot,
        "scatter": SpatialTrack,
    }
    if kind not in renderers:
        raise ValueError(f"kind must be one of {tuple(renderers)}")
    if colorbar not in ("shared", "each", False):
        raise ValueError("colorbar must be 'shared', 'each', or False.")
    if var_name is not None and isinstance(data, xr.Dataset):
        data = data[var_name]
    if isinstance(data, xr.Dataset) and not any(d == "variable" for d in (row, col)):
        if len(data.data_vars) != 1:
            raise ValueError("Specify var_name or facet by 'variable' for a Dataset.")
        data = data[next(iter(data.data_vars))]
    if observations is not None:
        _validate_observations(data, observations, obs_var)
    renderer = renderers[kind]
    if row is not None or col is not None:
        if kwargs.get("ax") is not None:
            raise ValueError("ax cannot be combined with faceting.")
        result = SpatialFacetGridPlot(
            data,
            row=row,
            col=col,
            col_wrap=col_wrap,
            colorbar=colorbar,
            _plotter_class=renderer,
            **kwargs,
        )
        result.map_monet(renderer, x=x, y=y, var_name=var_name)
        panels = list(result._panels())
        artists = list(result.mappables)
    else:
        if col_wrap is not None:
            raise ValueError("col_wrap requires col faceting.")
        if isinstance(data, pd.DataFrame):
            if var_name is None:
                raise ValueError("Specify var_name for tabular spatial data.")
            x, y = identify_coords(data, x, y)
            if kind == "scatter":
                data = xr.DataArray(
                    data[var_name].to_numpy(),
                    dims="sample",
                    coords={
                        x: ("sample", data[x].to_numpy()),
                        y: ("sample", data[y].to_numpy()),
                    },
                )
            else:
                data = data.set_index([y, x]).to_xarray()[var_name]
        if "size" in kwargs or "aspect" in kwargs:
            size = kwargs.pop("size", 4)
            aspect = kwargs.pop("aspect", 1.2)
            kwargs.setdefault("figsize", (size * aspect, size))
        result = renderer(data, lon_coord=x, lat_coord=y, **kwargs)
        result.plot(add_colorbar=colorbar is not False)
        panels = [(result.ax, data, {})]
        artists = [result.mappable]
    result.overlays = []
    for index, ((ax, subset, selection), artist) in enumerate(zip(panels, artists)):
        if observations is not None:
            obs = observations
            if isinstance(obs, (xr.DataArray, xr.Dataset)):
                obs = obs.sel({k: v for k, v in selection.items() if k in obs.dims})
            elif isinstance(obs, pd.DataFrame):
                for key, value in selection.items():
                    if key in obs.columns:
                        obs = obs.loc[obs[key] == value]
            ox, oy = identify_coords(obs)
            values = _observation_values(obs, obs_var)
            overlay = ax.scatter(
                obs[ox],
                obs[oy],
                c=values,
                **{
                    "norm": artist.norm,
                    "cmap": artist.cmap,
                    "s": 24,
                    "edgecolors": "black",
                    "linewidths": 0.5,
                    "transform": obs_crs or ccrs.PlateCarree(),
                    **(obs_kwargs or {}),
                },
            )
            result.overlays.append(overlay)
        if panel_labels:
            ax.text(
                0.02,
                0.98,
                f"({_panel_letter(index)})",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontweight="bold",
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            )
    return result


def _panel_letter(index):
    label = ""
    while True:
        index, digit = divmod(index, 26)
        label = chr(97 + digit) + label
        if index == 0:
            return label
        index -= 1


def _observation_values(obs, var):
    if isinstance(obs, xr.DataArray):
        return obs
    if var is None:
        raise ValueError("Specify obs_var for tabular or Dataset observations.")
    return obs[var]


def _validate_observations(data, obs, var):
    values = _observation_values(obs, var)
    fields = data.data_vars.values() if isinstance(data, xr.Dataset) else [data]
    for field in fields:
        unit = field.attrs.get("units")
        obs_unit = values.attrs.get("units")
        if unit and obs_unit and unit != obs_unit:
            raise ValueError(
                "Field and observation units differ; convert units before overlaying."
            )


def compare_spatial(observations, model, **kwargs):
    """Render aligned observation/model/bias maps; see SpatialComparisonPlot.

    Use row='model' or row='season' for repeated three-column comparisons.
    Gridded and collocated station DataArrays are supported. Regridding, unit
    conversion and station/model interpolation must be performed explicitly.
    The returned plot exposes bias, paired_counts, field_norm and bias_norm.
    """
    from .plots.spatial_comparison import SpatialComparisonPlot

    result = SpatialComparisonPlot(observations, model, **kwargs)
    try:
        return result.plot()
    except Exception:
        result.close()
        raise
