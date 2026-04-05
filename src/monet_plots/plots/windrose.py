from __future__ import annotations

import typing as t

import numpy as np
import pandas as pd

from ..plot_utils import normalize_data
from .base import BasePlot

if t.TYPE_CHECKING:
    import matplotlib.axes


class Windrose(BasePlot):
    """Windrose plot."""

    def __init__(self, *, wd: t.Any, ws: t.Any, **kwargs: t.Any) -> None:
        """
        Initialize Windrose Plot.

        Parameters
        ----------
        wd : Any
            Wind direction data. Can be numpy.ndarray, xarray.DataArray,
            or pandas.Series.
        ws : Any
            Wind speed data. Can be numpy.ndarray, xarray.DataArray,
            or pandas.Series.
        **kwargs : Any
            Keyword arguments passed to the parent class.
        """
        if "subplot_kw" not in kwargs:
            kwargs["subplot_kw"] = {"projection": "polar"}
        elif "projection" not in kwargs["subplot_kw"]:
            kwargs["subplot_kw"]["projection"] = "polar"

        super().__init__(**kwargs)

        if self.ax is None:
            self.ax = self.fig.add_subplot(projection="polar")

        from matplotlib.projections.polar import PolarAxes

        if not isinstance(self.ax, PolarAxes):
            raise ValueError("Windrose plot requires a polar axis.")

        self.wd = normalize_data(wd)
        self.ws = normalize_data(ws)

    def plot(
        self,
        *,
        bins: int | np.ndarray = 16,
        rose_bins: int | np.ndarray = 5,
        **kwargs: t.Any,
    ) -> matplotlib.axes.Axes:
        """
        Generate the windrose plot (Track A).

        Parameters
        ----------
        bins : int or numpy.ndarray, optional
            Number of bins for the wind direction or specific bin edges.
            Defaults to 16.
        rose_bins : int or numpy.ndarray, optional
            Number of bins for the wind speed or specific bin edges.
            Defaults to 5.
        **kwargs : Any
            Keyword arguments passed to `matplotlib.axes.Axes.bar`.

        Returns
        -------
        matplotlib.axes.Axes
            The polar axes object with the windrose plot.
        """
        from ..plot_utils import _update_history

        # Provenance tracking
        _update_history(self.wd, "Plotted Windrose (direction)")
        _update_history(self.ws, "Plotted Windrose (speed)")

        # Handle bins
        if isinstance(bins, int):
            bins = np.linspace(0, 360, bins + 1)

        if isinstance(rose_bins, int):
            # We need min/max. For lazy data, this triggers a compute if not provided.
            if hasattr(self.ws, "chunks"):
                import dask

                ws_min, ws_max = dask.compute(self.ws.min(), self.ws.max())
            else:
                ws_min, ws_max = self.ws.min(), self.ws.max()

            # Handle cases where min/max might be a pandas Series or xarray object
            ws_min = float(np.min(np.asarray(ws_min)))
            ws_max = float(np.max(np.asarray(ws_max)))
            rose_bins = np.linspace(ws_min, ws_max, rose_bins + 1)

        # Vectorized binning using histogram2d to support Dask
        if hasattr(self.wd, "chunks") or hasattr(self.ws, "chunks"):
            import dask.array as da

            wd_data = (
                self.wd.data if hasattr(self.wd, "data") else self.wd
            ).ravel()
            ws_data = (
                self.ws.data if hasattr(self.ws, "data") else self.ws
            ).ravel()
            h, _, _ = da.histogram2d(wd_data, ws_data, bins=[bins, rose_bins])
            h = h.compute()
        else:
            wd_data = np.asarray(self.wd).ravel()
            ws_data = np.asarray(self.ws).ravel()
            h, _, _ = np.histogram2d(wd_data, ws_data, bins=[bins, rose_bins])

        # Normalize to percentage of total observations
        total = h.sum()
        if total > 0:
            h = h / total

        # Cumulative sum along the speed dimension (axis 1)
        h_cumsum = np.cumsum(h, axis=1)

        # Labels and angles
        angles = np.deg2rad(bins[:-1])
        num_bins = bins.size - 1
        width = 2 * np.pi / num_bins

        self.ax.set_theta_zero_location("N")
        self.ax.set_theta_direction(-1)

        # Set radial grids
        r_max = h_cumsum[:, -1].max()
        r_ticks = np.linspace(0.2, 1.0, 5) * r_max
        self.ax.set_rgrids(
            r_ticks,
            labels=np.round(np.linspace(0.2, 1.0, 5) * 100).astype(int),
            angle=180,
        )

        # Plot bars from largest speed to smallest for stacking
        for i in range(h_cumsum.shape[1] - 1, -1, -1):
            label = f"{rose_bins[i]:.1f} - {rose_bins[i+1]:.1f}"
            self.ax.bar(
                angles + width / 2,
                h_cumsum[:, i],
                width=width,
                label=label,
                **kwargs,
            )

        self.ax.legend(title="Wind Speed", loc="lower left", bbox_to_anchor=(1.1, 0))
        return self.ax

    def hvplot(
        self,
        *,
        bins: int | np.ndarray = 16,
        rose_bins: int | np.ndarray = 5,
        **kwargs: t.Any,
    ) -> t.Any:
        """
        Generate an interactive windrose plot (Track B).

        Parameters
        ----------
        bins : int or numpy.ndarray, optional
            Number of bins for the wind direction or specific bin edges.
            Defaults to 16.
        rose_bins : int or numpy.ndarray, optional
            Number of bins for the wind speed or specific bin edges.
            Defaults to 5.
        **kwargs : Any
            Keyword arguments passed to `hvplot.bar`.

        Returns
        -------
        holoviews.core.Element
            The interactive windrose chart.
        """
        try:
            import hvplot.pandas  # noqa: F401
        except ImportError:
            raise ImportError(
                "hvplot and holoviews are required for interactive plotting."
            )

        # Handle bins (logic mirrored from plot for consistency)
        if isinstance(bins, int):
            bins = np.linspace(0, 360, bins + 1)

        if isinstance(rose_bins, int):
            if hasattr(self.ws, "chunks"):
                import dask

                ws_min, ws_max = dask.compute(self.ws.min(), self.ws.max())
            else:
                ws_min, ws_max = self.ws.min(), self.ws.max()

            # Handle cases where min/max might be a pandas Series or xarray object
            ws_min = float(np.min(np.asarray(ws_min)))
            ws_max = float(np.max(np.asarray(ws_max)))
            rose_bins = np.linspace(ws_min, ws_max, rose_bins + 1)

        # Vectorized binning using histogram2d
        if hasattr(self.wd, "chunks") or hasattr(self.ws, "chunks"):
            import dask.array as da

            wd_data = (
                self.wd.data if hasattr(self.wd, "data") else self.wd
            ).ravel()
            ws_data = (
                self.ws.data if hasattr(self.ws, "data") else self.ws
            ).ravel()
            h, _, _ = da.histogram2d(wd_data, ws_data, bins=[bins, rose_bins])
            h = h.compute()
        else:
            wd_data = np.asarray(self.wd).ravel()
            ws_data = np.asarray(self.ws).ravel()
            h, _, _ = np.histogram2d(wd_data, ws_data, bins=[bins, rose_bins])

        # Normalize to percentage
        total = h.sum()
        if total > 0:
            h = (h / total) * 100

        # Create DataFrame for hvplot
        angles = bins[:-1]
        speed_labels = [
            f"{rose_bins[i]:.1f}-{rose_bins[i+1]:.1f}"
            for i in range(len(rose_bins) - 1)
        ]

        df_list = []
        for i, label in enumerate(speed_labels):
            temp_df = pd.DataFrame(
                {"angle": angles, "percentage": h[:, i], "speed": label}
            )
            df_list.append(temp_df)

        df = pd.concat(df_list)

        # Sort by speed to ensure correct stacking order in legend/plot if needed
        # but hvplot handles 'stack=True' and 'by' well.

        return df.hvplot.bar(
            x="angle",
            y="percentage",
            by="speed",
            stacked=True,
            polar=True,
            xlabel="Direction (deg)",
            ylabel="Frequency (%)",
            **kwargs,
        )
