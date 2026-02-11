from __future__ import annotations

from typing import TYPE_CHECKING, Any, Union

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from .. import taylordiagram as td
from ..plot_utils import normalize_data
from .base import BasePlot

if TYPE_CHECKING:
    import matplotlib.figure

    from ..taylordiagram import TaylorDiagram


class TaylorDiagramPlot(BasePlot):
    """Create a Taylor diagram from Xarray, Dask, or Pandas data.

    A convenience wrapper for creating Taylor diagrams that summarize
    model performance using correlation, standard deviation, and RMS error.
    Supports lazy evaluation and provenance tracking.
    """

    def __init__(
        self,
        df: Any,
        col1: str = "obs",
        col2: str | list[str] = "model",
        label1: str = "OBS",
        scale: float = 1.5,
        dia: TaylorDiagram | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the plot with data and diagram settings.

        Parameters
        ----------
        df : Any
            The input data. Can be an xarray Dataset, DataArray, pandas
            DataFrame, or numpy ndarray. Preferred format is xarray with
            dask backing for large datasets.
        col1 : str, default 'obs'
            Column or variable name for observations (the reference).
        col2 : str or list of str, default 'model'
            Column or variable name(s) for model predictions (the samples).
        label1 : str, default 'OBS'
            Label for the reference (observation) point on the diagram.
        scale : float, default 1.5
            Scale factor for the diagram axes, relative to the reference
            standard deviation.
        dia : TaylorDiagram, optional
            An existing TaylorDiagram object to add samples to. If None,
            a new one is created.
        *args : Any
            Additional positional arguments passed to BasePlot.
        **kwargs : Any
            Additional keyword arguments passed to BasePlot.
        """
        super().__init__(*args, **kwargs)
        # We don't initialize self.ax here because TaylorDiagram creates its own

        self.df = normalize_data(df)
        self.col1 = col1
        self.col2 = [col2] if isinstance(col2, str) else col2

        # Verify columns/variables exist in the normalized data
        if isinstance(self.df, (xr.DataArray, xr.Dataset)):
            available = (
                list(self.df.coords) + list(self.df.data_vars)
                if isinstance(self.df, xr.Dataset)
                else [self.df.name] + list(self.df.coords)
            )
            for col in [self.col1] + self.col2:
                if col not in available and col is not None:
                    # If it's a DataArray and col matches the name, it's fine
                    if isinstance(self.df, xr.DataArray) and (
                        col == self.df.name or self.df.name is None
                    ):
                        continue
                    raise ValueError(f"Variable '{col}' not found in input data.")
        else:
            # Pandas
            for col in [self.col1] + self.col2:
                if col not in self.df.columns:
                    raise ValueError(f"Column '{col}' not found in DataFrame.")

        self.label1 = label1
        self.scale = scale
        self.dia = dia

        # Update history for provenance if xarray
        if isinstance(self.df, (xr.DataArray, xr.Dataset)):
            from ..verification_metrics import _update_history

            _update_history(self.df, "Initialized TaylorDiagramPlot")

    def plot(self, **kwargs: Any) -> td.TaylorDiagram:
        """Generate the Taylor diagram by computing statistics and plotting.

        This method calculates the standard deviations and correlations
        for the specified variables. For Dask-backed data, all statistics
        are computed in a single parallel operation to maximize efficiency.

        Parameters
        ----------
        **kwargs : Any
            Additional keyword arguments passed to `dia.add_sample`
            (and eventually to `matplotlib.axes.Axes.plot`). Common options
            include `marker`, `markersize`, `alpha`, and `ls`.

        Returns
        -------
        TaylorDiagram
            The TaylorDiagram object containing the plot.
        """
        from .. import verification_metrics

        # 1. Prepare lazy/vectorized statistics
        stats_to_compute = {}

        # Reference standard deviation (Observations)
        obs = self.df[self.col1]
        stats_to_compute["obs_std"] = obs.std()

        # Model statistics
        for model_col in self.col2:
            mod = self.df[model_col]
            stats_to_compute[f"{model_col}_std"] = mod.std()
            stats_to_compute[f"{model_col}_corr"] = verification_metrics.compute_corr(
                obs, mod
            )

        # 2. Batch computation if using Dask
        import dask

        computed_stats = dask.compute(stats_to_compute)[0]

        # 3. Initialize or Update Diagram
        if self.dia is None:
            obsstd = float(computed_stats["obs_std"])

            # Remove default axes created by BasePlot to avoid extra empty plot
            if hasattr(self, "ax") and self.ax is not None:
                self.fig.delaxes(self.ax)

            self.dia = td.TaylorDiagram(
                obsstd, scale=self.scale, fig=self.fig, rect=111, label=self.label1
            )
            self.ax = self.dia._ax

            # Add RMS contours
            contours = self.dia.add_contours(colors="0.5")
            plt.clabel(contours, inline=1, fontsize=10)

        # 4. Add samples to the diagram
        for model_col in self.col2:
            model_std = float(computed_stats[f"{model_col}_std"])
            cc = float(computed_stats[f"{model_col}_corr"])
            self.dia.add_sample(model_std, cc, label=model_col, **kwargs)

        # 5. Finalize UI
        self.fig.legend(
            self.dia.samplePoints,
            [p.get_label() for p in self.dia.samplePoints],
            numpoints=1,
            loc="upper right",
        )
        self.fig.tight_layout()

        # Update provenance
        if isinstance(self.df, (xr.DataArray, xr.Dataset)):
            verification_metrics._update_history(
                self.df, "Generated TaylorDiagramPlot"
            )

        return self.dia
