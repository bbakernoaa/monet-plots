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
            contour_kwargs = {'levels': np.arange(0, 1.1, 0.1),
                              'cmap': 'viridis'}
        if barb_kwargs is None:
            barb_kwargs = {}

        class GridObj:
            def __init__(self, lat, lon):
                self.variables = {'LAT': lat[np.newaxis, np.newaxis, ...],
                                  'LON': lon[np.newaxis, np.newaxis, ...]}

        lon2d, lat2d = np.meshgrid(self.lon, self.lat)
        gridobj = GridObj(lat2d, lon2d)

        contour_plot = SpatialContourPlot(
            modelvar=self.hgt,
            gridobj=gridobj,
            date=self.date,
            ax=self.ax,
            fig=self.fig
        )
        contour_plot.plot(**contour_kwargs)

        from .. import tools
        ws, wdir = tools.uv2wsdir(self.u, self.v)
        barb_plot = WindBarbsPlot(
            ws=ws,
            wdir=wdir,
            gridobj=gridobj,
            ax=self.ax,
            fig=self.fig
        )
        barb_plot.plot(**barb_kwargs)
