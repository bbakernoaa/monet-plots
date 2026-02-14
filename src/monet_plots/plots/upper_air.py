from __future__ import annotations

import typing as t

import numpy as np

from .spatial import SpatialPlot
from .spatial_contour import SpatialContourPlot
from .wind_barbs import WindBarbsPlot


class UpperAir(SpatialPlot):
    """Upper air chart plot."""

    def __init__(
        self,
        *,
        lat: np.ndarray,
        lon: np.ndarray,
        hgt: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        date: t.Any | None = None,
        **kwargs: t.Any,
    ) -> None:
        """
        Parameters
        ----------
        lat
            Latitude values.
        lon
            Longitude values.
        hgt
            Geopotential height values.
        u
            U-component of wind.
        v
            V-component of wind.
        date
            The date of the plot.
        **kwargs
            Keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self.lat = lat
        self.lon = lon
        self.hgt = hgt
        self.u = u
        self.v = v
        self.date = date

    def plot(
        self,
        *,
        contour_kwargs: dict | None = None,
        barb_kwargs: dict | None = None,
        **kwargs: t.Any,
    ) -> None:
        """
        Parameters
        ----------
        contour_kwargs
            Keyword arguments passed to `SpatialContourPlot.plot`.
        barb_kwargs
            Keyword arguments passed to `WindBarbsPlot.plot`.
        **kwargs
            Keyword arguments passed to the parent `plot` method.
        """
        if contour_kwargs is None:
            # Use data-driven levels for geopotential height
            # Typical 500 hPa heights range from ~5000-6000 m
            hgt_min, hgt_max = np.nanmin(self.hgt), np.nanmax(self.hgt)
            hgt_range = hgt_max - hgt_min
            if hgt_range > 0:
                # Create reasonable contour levels based on data range
                num_levels = 10
                levels = np.linspace(hgt_min, hgt_max, num_levels)
            else:
                # Fallback if data has no range
                levels = np.arange(hgt_min - 50, hgt_min + 51, 10)
            contour_kwargs = {"levels": levels, "cmap": "viridis"}
        if barb_kwargs is None:
            barb_kwargs = {}

        class GridObj:
            def __init__(self, lat, lon):
                self.variables = {
                    "LAT": lat[np.newaxis, np.newaxis, ...],
                    "LON": lon[np.newaxis, np.newaxis, ...],
                }

        lon2d, lat2d = np.meshgrid(self.lon, self.lat)
        gridobj = GridObj(lat2d, lon2d)

        contour_plot = SpatialContourPlot(
            modelvar=self.hgt, gridobj=gridobj, date=self.date, ax=self.ax, fig=self.fig
        )
        contour_plot.plot(**contour_kwargs)

        from .. import tools

        ws, wdir = tools.uv2wsdir(self.u, self.v)
        barb_plot = WindBarbsPlot(
            ws=ws, wdir=wdir, gridobj=gridobj, ax=self.ax, fig=self.fig
        )
        barb_plot.plot(**barb_kwargs)

    def hvplot(self, **kwargs: t.Any):
        """Generate an interactive upper air plot using hvPlot."""
        import hvplot.pandas  # noqa: F401
        import hvplot.xarray  # noqa: F401
        import xarray as xr

        lon2d, lat2d = np.meshgrid(self.lon, self.lat)
        da_hgt = xr.DataArray(
            self.hgt,
            dims=["lat", "lon"],
            coords={"lat": self.lat, "lon": self.lon},
            name="height",
        )
        da_u = xr.DataArray(
            self.u,
            dims=["lat", "lon"],
            coords={"lat": self.lat, "lon": self.lon},
            name="u",
        )
        da_v = xr.DataArray(
            self.v,
            dims=["lat", "lon"],
            coords={"lat": self.lat, "lon": self.lon},
            name="v",
        )

        hgt_plot = da_hgt.hvplot.contour(geo=True, title="Upper Air Height and Wind")

        # Calculate angle and mag for vectorfield
        da_angle = xr.apply_ufunc(np.arctan2, da_v, da_u)
        da_mag = xr.apply_ufunc(np.hypot, da_u, da_v)

        vector_plot = xr.Dataset(
            {"angle": da_angle, "mag": da_mag},
            coords={"lat": self.lat, "lon": self.lon},
        ).hvplot.vectorfield(x="lon", y="lat", angle="angle", mag="mag", geo=True)

        return hgt_plot * vector_plot
