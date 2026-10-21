# src/monet_plots/plots/spatial_overlay.py
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

from ..geo import identify_coords, variable_label
from ..plot_utils import _update_history, normalize_data
from .spatial import SpatialPlot

if TYPE_CHECKING:
    from matplotlib.axes import Axes


class SpatialOverlayPlot(SpatialPlot):
    """Plot gridded model data and overlay observation points on a map.

    This class supports plotting a 2D field (via contourf or imshow) and
    overlaying observation sites as scatter points, using a shared colorscale.
    """

    def __init__(
        self,
        model_data: Any,
        obs_data: Any,
        model_var: Optional[str] = None,
        obs_var: Optional[str] = None,
        *,
        fig: plt.Figure | None = None,
        ax: Axes | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the spatial overlay plot.

        Parameters
        ----------
        model_data : Any
            Gridded model data (xarray DataArray/Dataset).
        obs_data : Any
            Observation point data (Pandas DataFrame, Xarray Dataset).
        model_var : str
            Variable name in model_data.
        obs_var : str
            Variable name in obs_data.
        **kwargs : Any
            Additional arguments passed to SpatialPlot.
        """
        super().__init__(fig=fig, ax=ax, **kwargs)
        self.model_data = normalize_data(model_data)
        self.obs_data = normalize_data(obs_data)

        if model_var is None and isinstance(self.model_data, xr.DataArray):
            self.model_var = self.model_data.name
        elif (
            model_var is None
            and isinstance(self.model_data, xr.Dataset)
            and len(self.model_data.data_vars) == 1
        ):
            self.model_var = next(iter(self.model_data.data_vars))
        else:
            self.model_var = model_var

        if obs_var is None:
            # Attempt to identify variable that is not a coordinate
            if isinstance(self.obs_data, xr.DataArray):
                self.obs_var = self.obs_data.name
            elif hasattr(self.obs_data, "data_vars"):
                vars = list(self.obs_data.data_vars)
                if vars:
                    self.obs_var = vars[0]
            elif hasattr(self.obs_data, "columns"):
                # Simple heuristic for pandas
                coords = [
                    "lat",
                    "lon",
                    "latitude",
                    "longitude",
                    "x",
                    "y",
                    "time",
                    "datetime",
                ]
                vars = [
                    c
                    for k in self.obs_data.columns
                    if (c := str(k)).lower() not in coords
                ]
                if vars:
                    self.obs_var = vars[0]
        else:
            self.obs_var = obs_var

        if isinstance(self.model_data, xr.Dataset) and self.model_var is None:
            raise ValueError("Specify model_var for a Dataset with multiple variables.")
        if not hasattr(self, "obs_var"):
            raise ValueError("Specify obs_var for observation data.")

        _update_history(
            self.model_data, "Initialized monet-plots.SpatialOverlayPlot (model)"
        )
        _update_history(
            self.obs_data, "Initialized monet-plots.SpatialOverlayPlot (obs)"
        )

    def plot(
        self,
        kind: str = "contourf",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cmap: str = "viridis",
        nlevels: int = 20,
        **kwargs: Any,
    ) -> Axes:
        """Generate the spatial overlay plot.

        Parameters
        ----------
        kind : str
            Type of gridded plot: 'contourf' or 'imshow'.
        vmin, vmax : float, optional
            Common min/max for the colorscale. If None, calculated from data.
        cmap : str
            Colormap to use.
        nlevels : int
            Number of levels for contourf.
        **kwargs : Any
            Additional arguments for the gridded plot.
        """
        from .facet_grid import _shared_scale
        from .spatial_contour import SpatialContourPlot
        from .spatial_imshow import SpatialImshowPlot
        from .spatial_mesh import SpatialMeshPlot

        renderers = {
            "contourf": SpatialContourPlot,
            "imshow": SpatialImshowPlot,
            "pcolormesh": SpatialMeshPlot,
        }
        if kind not in renderers:
            raise ValueError("kind must be 'contourf', 'imshow', or 'pcolormesh'")
        options = {**self._plot_defaults, **kwargs}
        add_colorbar = options.pop("add_colorbar", True)
        cbar_kwargs = options.pop("cbar_kwargs", {})
        obs_crs = options.pop("obs_crs", ccrs.PlateCarree())
        model = (
            self.model_data[self.model_var]
            if isinstance(self.model_data, xr.Dataset)
            else self.model_data
        )
        obs = (
            self.obs_data[self.obs_var]
            if isinstance(self.obs_data, (xr.Dataset, pd.DataFrame))
            else self.obs_data
        )
        if (
            model.attrs.get("units")
            and obs.attrs.get("units")
            and model.attrs["units"] != obs.attrs["units"]
        ):
            raise ValueError(
                "Model and observation units differ; convert units before overlaying."
            )
        if options.get("norm") is None:
            import dask

            bounds = dask.compute(model.min(), model.max(), obs.min(), obs.max())
            vmin = min(float(bounds[0]), float(bounds[2])) if vmin is None else vmin
            vmax = max(float(bounds[1]), float(bounds[3])) if vmax is None else vmax
            options.update(vmin=vmin, vmax=vmax)
        options.setdefault("cmap", cmap)
        if kind == "contourf":
            options.setdefault("levels", nlevels)
        options = _shared_scale(model, options, renderers[kind])
        options = self.add_features(**options)
        renderer = renderers[kind](
            model,
            ax=self.ax,
            data_crs=self.data_crs,
            extent_crs=self.extent_crs,
            coastlines=False,
            style=self._style,
        )
        renderer.plot(**options, add_colorbar=False)
        self.mappable = renderer.mappable
        lon_o, lat_o = identify_coords(self.obs_data)
        self.observation_artist = self.ax.scatter(
            self.obs_data[lon_o],
            self.obs_data[lat_o],
            c=obs,
            norm=self.mappable.norm,
            cmap=self.mappable.cmap,
            edgecolors="k",
            s=30,
            transform=obs_crs,
            label="Observations",
        )
        if add_colorbar:
            self.add_colorbar(
                self.mappable, **{"label": variable_label(model), **cbar_kwargs}
            )
        return self.ax

    def hvplot(self, **kwargs: Any) -> Any:
        """Generate an interactive spatial overlay plot using hvPlot (Track B)."""
        try:
            import hvplot.pandas  # noqa: F401
            import hvplot.xarray  # noqa: F401
        except ImportError:
            raise ImportError(
                "hvplot is required for interactive plotting. Install it with 'pip install hvplot'."
            )

        lon_m, lat_m = self._identify_coords(self.model_data)
        lon_o, lat_o = self._identify_coords(self.obs_data)

        plot_m = self.model_data.hvplot(
            x=lon_m, y=lat_m, geo=True, rasterize=True, cmap="viridis", **kwargs
        )

        plot_o = self.obs_data.hvplot.scatter(
            x=lon_o, y=lat_o, c=self.obs_var, geo=True, cmap="viridis", **kwargs
        )

        return plot_m * plot_o
