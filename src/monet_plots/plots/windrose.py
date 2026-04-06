from __future__ import annotations

import typing as t

import numpy as np
import pandas as pd
import xarray as xr

from ..plot_utils import _update_history, compute, is_dask, is_lazy, normalize_data
from .base import BasePlot

if t.TYPE_CHECKING:
    from matplotlib.axes import Axes


class Windrose(BasePlot):
    """Windrose plot class for static and interactive visualizations."""

    def __init__(
        self,
        *,
        wd: np.ndarray | xr.DataArray | pd.Series,
        ws: np.ndarray | xr.DataArray | pd.Series,
        **kwargs: t.Any,
    ) -> None:
        """
        Initialize Windrose Plot.

        Parameters
        ----------
        wd : numpy.ndarray or xarray.DataArray or pandas.Series
            Wind direction data (degrees, 0-360).
        ws : numpy.ndarray or xarray.DataArray or pandas.Series
            Wind speed data.
        **kwargs : Any
            Keyword arguments passed to the parent class.

        Examples
        --------
        >>> import numpy as np
        >>> wd = np.random.uniform(0, 360, 100)
        >>> ws = np.random.uniform(0, 20, 100)
        >>> wr = Windrose(wd=wd, ws=ws)
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

    def _compute_windrose_histogram(
        self,
        *,
        bins: int | np.ndarray = 16,
        rose_bins: int | np.ndarray = 5,
        percentage: bool = False,
    ) -> t.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the 2D histogram for the windrose.

        Parameters
        ----------
        bins : int or numpy.ndarray, optional
            Number of bins for the wind direction or specific bin edges.
            Defaults to 16.
        rose_bins : int or numpy.ndarray, optional
            Number of bins for the wind speed or specific bin edges.
            Defaults to 5.
        percentage : bool, optional
            If True, returns histogram normalized to 0-100 percentage.
            If False, returns decimal frequency (0-1).
            Defaults to False.

        Returns
        -------
        h : numpy.ndarray
            The 2D histogram (normalized).
        dir_edges : numpy.ndarray
            The direction bin edges.
        speed_edges : numpy.ndarray
            The speed bin edges.

        Examples
        --------
        >>> h, dir_edges, speed_edges = wr._compute_windrose_histogram(bins=8)
        """
        # Handle direction bins
        if isinstance(bins, int):
            dir_edges = np.linspace(0, 360, bins + 1)
        else:
            dir_edges = bins

        # Handle speed bins
        if isinstance(rose_bins, int):
            # We need min/max. For lazy data, this triggers a compute if not provided.
            ws_min, ws_max = compute(self.ws.min(), self.ws.max())

            # Handle cases where min/max might be a pandas Series or xarray object
            ws_min = float(np.min(np.asarray(ws_min)))
            ws_max = float(np.max(np.asarray(ws_max)))
            speed_edges = np.linspace(ws_min, ws_max, rose_bins + 1)
        else:
            speed_edges = rose_bins

        # Vectorized binning using histogram2d to support Dask
        if is_lazy(self.wd) or is_lazy(self.ws):
            wd_data = (self.wd.data if hasattr(self.wd, "data") else self.wd).ravel()
            ws_data = (self.ws.data if hasattr(self.ws, "data") else self.ws).ravel()

            if is_dask(self.wd) or is_dask(self.ws):
                import dask.array as da

                h, _, _ = da.histogram2d(
                    wd_data, ws_data, bins=[dir_edges, speed_edges]
                )
                h = compute(h)
            else:
                # Fallback for Cubed or other lazy backends that lack histogram2d.
                # Compute and use numpy.histogram2d.
                wd_eager, ws_eager = compute(wd_data, ws_data)
                h, _, _ = np.histogram2d(
                    wd_eager, ws_eager, bins=[dir_edges, speed_edges]
                )
        else:
            wd_data = np.asarray(self.wd).ravel()
            ws_data = np.asarray(self.ws).ravel()
            h, _, _ = np.histogram2d(wd_data, ws_data, bins=[dir_edges, speed_edges])

        # Normalize
        total = h.sum()
        if total > 0:
            h = h / total
            if percentage:
                h = h * 100

        return h, dir_edges, speed_edges

    def plot(
        self,
        *,
        bins: int | np.ndarray = 16,
        rose_bins: int | np.ndarray = 5,
        **kwargs: t.Any,
    ) -> Axes:
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

        Examples
        --------
        >>> ax = wr.plot(bins=16, rose_bins=5)
        """
        # Provenance tracking
        _update_history(self.wd, "Plotted Windrose (direction)")
        _update_history(self.ws, "Plotted Windrose (speed)")

        # Compute histogram
        h, dir_edges, speed_edges = self._compute_windrose_histogram(
            bins=bins, rose_bins=rose_bins, percentage=False
        )

        # Cumulative sum along the speed dimension (axis 1) for stacking
        h_cumsum = np.cumsum(h, axis=1)

        # Labels and angles
        angles = np.deg2rad(dir_edges[:-1])
        num_bins = dir_edges.size - 1
        width = 2 * np.pi / num_bins

        self.ax.set_theta_zero_location("N")
        self.ax.set_theta_direction(-1)

        # Set radial grids correctly as percentages
        r_max = h_cumsum[:, -1].max()
        r_ticks = np.linspace(0.2, 1.0, 5) * r_max
        r_labels = [f"{val * 100:.0f}%" for val in r_ticks]

        self.ax.set_rgrids(
            r_ticks,
            labels=r_labels,
            angle=180,
        )

        # Plot bars from largest speed to smallest for stacking
        for i in range(h_cumsum.shape[1] - 1, -1, -1):
            label = f"{speed_edges[i]:.1f} - {speed_edges[i + 1]:.1f}"
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
            Keyword arguments passed to `hvplot.scatter`.

        Returns
        -------
        holoviews.core.Element
            The interactive windrose chart.

        Examples
        --------
        >>> plot = wr.hvplot(bins=16, rose_bins=5)
        """
        try:
            import hvplot.pandas  # noqa: F401
        except ImportError:
            raise ImportError(
                "hvplot and holoviews are required for interactive plotting."
            )

        # Provenance tracking
        _update_history(self.wd, "Plotted Windrose (direction)")
        _update_history(self.ws, "Plotted Windrose (speed)")

        # Compute histogram (Track B uses percentage 0-100)
        h, dir_edges, speed_edges = self._compute_windrose_histogram(
            bins=bins, rose_bins=rose_bins, percentage=True
        )

        # Create DataFrame for HoloViews/hvplot
        angles = dir_edges[:-1]
        speed_labels = [
            f"{speed_edges[i]:.1f}-{speed_edges[i + 1]:.1f}"
            for i in range(len(speed_edges) - 1)
        ]

        df_list = []
        for i, label in enumerate(speed_labels):
            temp_df = pd.DataFrame(
                {"angle": angles, "percentage": h[:, i], "speed": label}
            )
            df_list.append(temp_df)

        df = pd.concat(df_list)

        # Convert angles from degrees to radians for HoloViews/Bokeh polar plots
        df["angle"] = np.deg2rad(df["angle"])

        # Use scatter with polar=True as it is well-supported for polar charts.
        # Note: True polar 'bars' are difficult in hvPlot without direct HoloViews,
        # but a polar scatter/line provides a clear alternative for Track B.
        return df.hvplot.scatter(
            x="angle",
            y="percentage",
            by="speed",
            polar=True,
            xlabel="Direction (deg)",
            ylabel="Frequency (%)",
            **kwargs,
        )
