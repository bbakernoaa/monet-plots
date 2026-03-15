# src/monet_plots/plots/spatial_overlay.py
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Optional

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from ..plot_utils import _update_history, get_plot_kwargs, normalize_data
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
        else:
            self.model_var = model_var

        self.obs_var = obs_var

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
        # Coordinate identification for model
        lon_m, lat_m = self._identify_coords(self.model_data)
        # Coordinate identification for obs
        lon_o, lat_o = self._identify_coords(self.obs_data)

        # Determine common vmin/vmax
        if vmin is None or vmax is None:
            m_da = (
                self.model_data[self.model_var]
                if isinstance(self.model_data, xr.Dataset)
                else self.model_data
            )
            o_da = (
                self.obs_data[self.obs_var]
                if isinstance(self.obs_data, (xr.Dataset, pd.DataFrame))
                else self.obs_data
            )

            m_min, m_max = m_da.min(), m_da.max()
            o_min, o_max = o_da.min(), o_da.max()

            # Efficiently compute if dask
            try:
                import dask

                vals = dask.compute(m_min, m_max, o_min, o_max)
                vmin = vmin or float(min(vals[0], vals[2]))
                vmax = vmax or float(max(vals[1], vals[3]))
            except (ImportError, AttributeError):
                vmin = vmin or min(float(m_min), float(o_min))
                vmax = vmax or max(float(m_max), float(o_max))

        # Setup plot kwargs
        plot_kwargs = self.add_features(**kwargs)
        plot_kwargs.setdefault("transform", ccrs.PlateCarree())

        levels = np.linspace(vmin, vmax, nlevels)
        final_gridded_kwargs = get_plot_kwargs(
            cmap=cmap, vmin=vmin, vmax=vmax, levels=levels, **plot_kwargs
        )

        # Plot gridded data
        m_da = (
            self.model_data[self.model_var]
            if isinstance(self.model_data, xr.Dataset)
            else self.model_data
        )
        if kind == "contourf":
            mappable = self.ax.contourf(
                self.model_data[lon_m],
                self.model_data[lat_m],
                m_da,
                **final_gridded_kwargs,
            )
        elif kind == "imshow":
            # Simplified imshow logic
            extent = [
                float(self.model_data[lon_m].min()),
                float(self.model_data[lon_m].max()),
                float(self.model_data[lat_m].min()),
                float(self.model_data[lat_m].max()),
            ]
            mappable = self.ax.imshow(
                m_da, extent=extent, origin="lower", **final_gridded_kwargs
            )
        else:
            raise ValueError("kind must be 'contourf' or 'imshow'")

        # Overlay obs
        self.ax.scatter(
            self.obs_data[lon_o],
            self.obs_data[lat_o],
            c=self.obs_data[self.obs_var],
            norm=mappable.norm,
            cmap=mappable.cmap,
            edgecolors="k",
            s=30,
            transform=ccrs.PlateCarree(),
            label="Observations",
        )

        self.add_colorbar(mappable)

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
