# src/monet_plots/plots/curtain.py
"""Vertical curtain plot for cross-sectional data."""

import xarray as xr
from .base import BasePlot
from ..plot_utils import get_plot_kwargs
from typing import Any, Optional


class CurtainPlot(BasePlot):
    """Vertical curtain plot for cross-sectional data.

    This plot shows a 2D variable (e.g., concentration) as a function of
    one horizontal dimension (time or distance) and one vertical dimension
    (altitude or pressure).
    """

    def __init__(
        self,
        data: Any,
        *,
        x: Optional[str] = None,
        y: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize Curtain Plot.

        Args:
            data: Input data. Should be a 2D xarray.DataArray or similar.
            x: Name of the x-axis dimension/coordinate (e.g., 'time').
            y: Name of the y-axis dimension/coordinate (e.g., 'level').
            **kwargs: Arguments passed to BasePlot.
        """
        super().__init__(**kwargs)
        self.data = data
        self.x = x
        self.y = y

    def plot(self, kind: str = "pcolormesh", colorbar: bool = True, **kwargs):
        """
        Generate the curtain plot.

        Args:
            kind: Type of plot ('pcolormesh' or 'contourf').
            colorbar: Whether to add a colorbar.
            **kwargs: Additional arguments for the plotting function.
        """
        plot_kwargs = get_plot_kwargs(**kwargs)
        plot_kwargs.setdefault("cmap", "viridis")

        # Ensure we have a DataArray
        if not isinstance(self.data, xr.DataArray):
            # Try to convert or at least verify it's xarray-like
            if hasattr(self.data, "to_array"):
                da = self.data.to_array()
            else:
                raise TypeError(
                    "CurtainPlot requires xarray-like data with 2 dimensions."
                )
        else:
            da = self.data

        if da.ndim != 2:
            raise ValueError(f"CurtainPlot requires 2D data, got {da.ndim}D.")

        # Determine x and y if not provided
        if self.x is None:
            self.x = da.dims[1]
        if self.y is None:
            self.y = da.dims[0]

        if kind == "pcolormesh":
            mappable = self.ax.pcolormesh(
                da[self.x], da[self.y], da, shading="auto", **plot_kwargs
            )
        elif kind == "contourf":
            mappable = self.ax.contourf(da[self.x], da[self.y], da, **plot_kwargs)
        else:
            raise ValueError("kind must be 'pcolormesh' or 'contourf'")

        if colorbar:
            # Auto-label colorbar from variable attrs
            _long_name = getattr(da, "attrs", {}).get("long_name", "")
            _units = getattr(da, "attrs", {}).get("units", "")
            _cbar_label = (
                f"{_long_name} ({_units})"
                if _long_name and _units
                else _long_name or _units or None
            )
            self.add_colorbar(mappable, label=_cbar_label)

        # Use long_name as axis labels when available, fall back to dim names
        self.ax.set_xlabel(
            da.coords[self.x].attrs.get("long_name", self.x)
            if self.x in da.coords
            else self.x
        )
        self.ax.set_ylabel(
            da.coords[self.y].attrs.get("long_name", self.y)
            if self.y in da.coords
            else self.y
        )

        # Invert y-axis for pressure-level data (hPa decreases with altitude)
        _y_units = ""
        if self.y in da.coords:
            _y_units = da.coords[self.y].attrs.get("units", "").lower()
        elif self.y in da.attrs:
            _y_units = str(da.attrs.get("units", "")).lower()
        if (
            any(u in _y_units for u in ["hpa", "mbar", "pa", "mb"])
            and not self.ax.yaxis_inverted()
        ):
            self.ax.invert_yaxis()

        # Title from variable attrs
        _title = da.attrs.get("long_name") or (da.name if da.name else None)
        if _title and not self.ax.get_title():
            self.ax.set_title(_title)

        return self.ax
