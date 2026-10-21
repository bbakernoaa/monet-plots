"""Paired observation/model/bias maps with scientifically comparable scales."""

from __future__ import annotations

import cartopy.crs as ccrs
import dask
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm

from ..geo import identify_coords, validate_data_crs, variable_label
from .base import BasePlot
from .spatial import SpatialTrack
from .spatial_contour import SpatialContourPlot
from .spatial_imshow import SpatialImshowPlot
from .spatial_mesh import SpatialMeshPlot


class SpatialComparisonPlot(BasePlot):
    """Observation, model and model-minus-observation panels, optionally by row.

    Inputs must be xarray DataArrays with the same dimension labels and spatial
    coordinates, already in the same units and CRS. A model-only ``row``
    dimension intentionally repeats static observations for each model/period.
    All other broadcasting and implicit regridding are rejected. Initialization
    preserves lazy field values; rendering resolves scales and computes panels.
    """

    def __init__(
        self,
        observations,
        model,
        *,
        row=None,
        kind="pcolormesh",
        x=None,
        y=None,
        projection=None,
        data_crs=None,
        extent_crs=None,
        size=3,
        aspect=1.25,
        style="wiley",
        **kwargs,
    ):
        if not isinstance(observations, xr.DataArray) or not isinstance(
            model, xr.DataArray
        ):
            raise TypeError(
                "Comparison inputs must be xarray DataArrays; select Dataset variables explicitly."
            )
        if kind not in {"pcolormesh", "imshow", "contourf", "scatter"}:
            raise ValueError("kind must be pcolormesh, imshow, contourf, or scatter.")
        self.x, self.y = identify_coords(model, x, y)
        self.observations, self.model = _paired_fields(
            observations,
            model,
            row=row,
            x=self.x,
            y=self.y,
        )
        validate_data_crs(model, self.x, self.y, data_crs)
        validate_data_crs(observations, self.x, self.y, data_crs)
        self.row, self.kind = row, kind
        field_dims = [dim for dim in model.dims if dim != row]
        expected = 1 if kind == "scatter" else 2
        if len(field_dims) != expected:
            raise ValueError(
                f"{kind} requires {expected} field dimensions; select extra dimensions or specify row."
            )
        self.paired = np.isfinite(self.model) & np.isfinite(self.observations)
        self.bias = (self.model - self.observations).where(self.paired)
        self.bias.name = "bias"
        self.bias.attrs = {"long_name": "Model − observations"}
        if model.attrs.get("units"):
            self.bias.attrs["units"] = model.attrs["units"]
        self.paired_counts = self.paired.sum(dim=field_dims)
        self.paired_counts.attrs = {
            "long_name": "Finite observation/model pairs",
            "units": "1",
        }
        self.data_crs = data_crs or ccrs.PlateCarree()
        self.extent_crs = extent_crs or ccrs.PlateCarree()
        self.render_options = kwargs
        self.plotters, self.mappables = [], []
        rows = model.sizes[row] if row else 1
        if rows == 0:
            raise ValueError("The comparison needs at least one row.")
        fig, axes = plt.subplots(
            rows,
            3,
            squeeze=False,
            figsize=(3 * size * aspect, rows * size),
            subplot_kw={"projection": projection or ccrs.PlateCarree()},
            layout="constrained",
        )
        super().__init__(fig=fig, ax=axes[0, 0], style=style)
        self.axes = axes

    def plot(
        self,
        *,
        vmin=None,
        vmax=None,
        norm=None,
        bias_limit=None,
        bias_norm=None,
        cmap="viridis",
        bias_cmap="RdBu_r",
        colorbar=True,
        cbar_kwargs=None,
        bias_cbar_kwargs=None,
        panel_labels=True,
        materialize="row",
        **kwargs,
    ):
        """Render with a common concentration scale and a zero-centered bias scale.

        Explicit vmin/vmax and bias_limit avoid global scale reductions. With
        automatic limits, all required scalar reductions share one Dask compute
        call. ``bias_norm`` must be a TwoSlopeNorm centered on zero; explicit
        ``bias_limit`` uses symmetric bounds. Empty panels retain masked values.
        """
        if self.mappables:
            raise RuntimeError(
                "This comparison has already been rendered; create a new plot to render again."
            )
        options = {**self.render_options, **kwargs}
        # Accept rendering configuration through the convenience constructor too.
        vmin, vmax = options.pop("vmin", vmin), options.pop("vmax", vmax)
        norm = options.pop("norm", norm)
        bias_limit = options.pop("bias_limit", bias_limit)
        bias_norm = options.pop("bias_norm", bias_norm)
        cmap, bias_cmap = options.pop("cmap", cmap), options.pop("bias_cmap", bias_cmap)
        colorbar = options.pop("colorbar", colorbar)
        cbar_kwargs = options.pop("cbar_kwargs", cbar_kwargs) or {}
        bias_cbar_kwargs = options.pop("bias_cbar_kwargs", bias_cbar_kwargs) or {}
        panel_labels = options.pop("panel_labels", panel_labels)
        materialize = options.pop("materialize", materialize)
        if materialize not in {"row", "panel"}:
            raise ValueError("materialize must be row or panel.")
        self.field_norm, self.bias_norm = _comparison_norms(
            self.observations,
            self.model,
            self.bias,
            vmin=vmin,
            vmax=vmax,
            norm=norm,
            bias_limit=bias_limit,
            bias_norm=bias_norm,
        )
        renderers = {
            "pcolormesh": SpatialMeshPlot,
            "imshow": SpatialImshowPlot,
            "contourf": SpatialContourPlot,
            "scatter": SpatialTrack,
        }
        feature_keys = {
            "extent",
            "coastlines",
            "states",
            "countries",
            "borders",
            "gridlines",
            "resolution",
            "natural_earth",
            "ocean",
            "land",
            "lakes",
            "rivers",
        }
        features = {
            key: options.pop(key) for key in list(options) if key in feature_keys
        }
        features.setdefault("coastlines", True)
        bias_levels = options.pop("bias_levels", None)
        for i in range(self.axes.shape[0]):
            fields = [
                data.isel({self.row: i}) if self.row and self.row in data.dims else data
                for data in (self.observations, self.model, self.bias)
            ]
            fields = [data.where(np.isfinite(data)) for data in fields]
            if materialize == "row" and any(data.chunks is not None for data in fields):
                # Shared Dask dependencies are evaluated once per three-panel row.
                # This bounds temporary data to a row rather than persisting the cube.
                fields = dask.compute(*fields)
            for j, (subset, title) in enumerate(
                zip(fields, ("Observations", "Model", "Model − observations"))
            ):
                selected_norm = self.bias_norm if j == 2 else self.field_norm
                panel_options = dict(options)
                if self.kind == "contourf":
                    levels = (
                        bias_levels if j == 2 else panel_options.pop("levels", None)
                    )
                    if levels is None or np.isscalar(levels):
                        levels = np.linspace(
                            selected_norm.vmin,
                            selected_norm.vmax,
                            int(levels) if levels is not None else 11,
                        )
                    panel_options["levels"] = levels
                plotter = renderers[self.kind](
                    subset,
                    ax=self.axes[i, j],
                    data_crs=self.data_crs,
                    extent_crs=self.extent_crs,
                    lon_coord=self.x,
                    lat_coord=self.y,
                    style=self._style,
                    **features,
                )
                plotter.plot(
                    norm=selected_norm,
                    cmap=bias_cmap if j == 2 else cmap,
                    add_colorbar=False,
                    **panel_options,
                )
                self.plotters.append(plotter)
                self.mappables.append(plotter.mappable)
                ax = self.axes[i, j]
                if self.row:
                    title = f"{self.row} = {self.model[self.row].values[i]}\n{title}"
                ax.set_title(title)
                if panel_labels:
                    from ..api import _panel_letter

                    ax.text(
                        0.02,
                        0.98,
                        f"({_panel_letter(i * 3 + j)})",
                        transform=ax.transAxes,
                        va="top",
                        fontweight="bold",
                        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
                    )
                if hasattr(plotter, "gridliner"):
                    grid = plotter.gridliner
                    if any(
                        (
                            grid.left_labels,
                            grid.right_labels,
                            grid.top_labels,
                            grid.bottom_labels,
                        )
                    ):
                        grid.top_labels = grid.right_labels = False
                        grid.left_labels = j == 0
                        grid.bottom_labels = i == self.axes.shape[0] - 1
        self.mappable = self.mappables[0]
        if colorbar:
            common = {"orientation": "horizontal", "pad": 0.08, **cbar_kwargs}
            self.colorbars.append(
                self.fig.colorbar(
                    self.mappables[0],
                    ax=list(self.axes[:, :2].flat),
                    **{"label": variable_label(self.model), **common},
                )
            )
            self.colorbars.append(
                self.fig.colorbar(
                    self.mappables[2],
                    ax=list(self.axes[:, 2].flat),
                    **{
                        "label": variable_label(self.bias),
                        **common,
                        **bias_cbar_kwargs,
                    },
                )
            )
        return self


def _paired_fields(observations, model, *, row, x, y):
    if not isinstance(observations, xr.DataArray) or not isinstance(
        model, xr.DataArray
    ):
        raise TypeError(
            "Comparison inputs must be xarray DataArrays; select Dataset variables explicitly."
        )
    if row is not None and row not in model.dims:
        raise ValueError(f"Row dimension {row!r} is not present in model data.")
    for name in (x, y):
        if name not in observations.coords or name not in model.coords:
            raise ValueError(
                f"Both fields must use the same spatial coordinate names; missing {name!r}."
            )
    expected_dims = set(model.dims) - {row}
    if set(observations.dims) - {row} != expected_dims:
        raise ValueError(
            "Observation/model field dimensions differ; collocate or regrid explicitly before comparison."
        )
    if not all(model.sizes[d] == observations.sizes[d] for d in expected_dims):
        raise ValueError(
            "Observation/model field sizes differ; collocate or regrid explicitly before comparison."
        )
    obs_unit, model_unit = observations.attrs.get("units"), model.attrs.get("units")
    if obs_unit != model_unit:
        raise ValueError(
            "Observation/model units differ or are missing on one input; convert and label units explicitly."
        )
    try:
        obs, mod = xr.align(observations, model, join="exact", copy=False)
    except ValueError as exc:
        raise ValueError(
            "Observation/model coordinate labels differ; align or collocate explicitly before comparison."
        ) from exc
    for name in (x, y):
        if name not in obs.coords or name not in mod.coords:
            raise ValueError(
                f"Both fields must use the same spatial coordinate names; missing {name!r}."
            )
        oc, mc = xr.broadcast(obs[name], mod[name])
        if not bool((oc == mc).all()):
            raise ValueError(
                f"Observation/model spatial coordinate {name!r} differs; regrid or collocate explicitly."
            )
    return obs.copy(deep=False), mod.copy(deep=False)


def _comparison_norms(obs, model, bias, *, vmin, vmax, norm, bias_limit, bias_norm):
    if norm is not None and (vmin is not None or vmax is not None):
        raise ValueError("Supply norm or vmin/vmax, not both.")
    field_norm = norm if norm is not None else Normalize(vmin, vmax)
    if bias_norm is not None and bias_limit is not None:
        raise ValueError("Supply bias_norm or bias_limit, not both.")
    if bias_norm is not None and (
        not isinstance(bias_norm, TwoSlopeNorm) or bias_norm.vcenter != 0
    ):
        raise ValueError("bias_norm must be a TwoSlopeNorm centered on zero.")
    if bias_limit is not None and (not np.isfinite(bias_limit) or bias_limit <= 0):
        raise ValueError("bias_limit must be finite and positive.")
    reductions = {}
    positive = isinstance(field_norm, LogNorm)
    valid_obs = obs.where(np.isfinite(obs) & ((obs > 0) if positive else True))
    valid_model = model.where(np.isfinite(model) & ((model > 0) if positive else True))
    if field_norm.vmin is None:
        reductions.update(obs_min=valid_obs.min(), model_min=valid_model.min())
    if field_norm.vmax is None:
        reductions.update(obs_max=valid_obs.max(), model_max=valid_model.max())
    needs_bias = bias_norm is None or bias_norm.vmin is None or bias_norm.vmax is None
    if needs_bias and bias_limit is None:
        reductions["bias_max"] = np.abs(bias).max()
    values = (
        dict(zip(reductions, map(float, dask.compute(*reductions.values()))))
        if reductions
        else {}
    )
    for bound, reducer in (("vmin", np.fmin), ("vmax", np.fmax)):
        if getattr(field_norm, bound) is None:
            key = "min" if bound == "vmin" else "max"
            value = reducer(values[f"obs_{key}"], values[f"model_{key}"])
            if not np.isfinite(value):
                raise ValueError(
                    "No finite values available for automatic concentration limits; supply explicit limits."
                )
            setattr(field_norm, bound, value)
    if field_norm.vmin == field_norm.vmax:
        delta = abs(field_norm.vmin) * 0.01 or 0.5
        field_norm.vmin -= delta
        field_norm.vmax += delta
    limit = bias_limit if bias_limit is not None else values.get("bias_max")
    if needs_bias:
        if not np.isfinite(limit):
            raise ValueError(
                "No finite paired values available for automatic bias limits; supply bias_limit to show empty panels."
            )
        limit = limit or 1.0  # An all-zero bias still needs a finite color scale.
    if bias_norm is None:
        bias_norm = TwoSlopeNorm(vcenter=0, vmin=-limit, vmax=limit)
    else:
        if bias_norm.vmin is None:
            bias_norm.vmin = -limit
        if bias_norm.vmax is None:
            bias_norm.vmax = limit
    return field_norm, bias_norm
