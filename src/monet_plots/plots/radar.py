# src/monet_plots/plots/radar.py
"""Radar (spider) chart for model evaluation."""

from __future__ import annotations

import numpy as np
import xarray as xr
from typing import Any, Optional, Union, List, TYPE_CHECKING

from .base import BasePlot
from ..plot_utils import _update_history, normalize_data
from ..verification_metrics import compute_radar_metrics

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure


class RadarPlot(BasePlot):
    """Radar (spider) chart for model evaluation.

    Visualizes multiple normalized performance metrics across one or more models.
    Metrics are typically normalized to a 0-1 scale.

    Attributes
    ----------
    metrics_data : xr.Dataset
        Dataset containing the normalized metrics for each model.
    """

    def __init__(
        self,
        data: Any = None,
        *,
        obs_col: Optional[str] = None,
        mod_cols: Optional[Union[str, List[str]]] = None,
        metrics: Optional[List[str]] = None,
        metrics_data: Optional[xr.Dataset] = None,
        fig: Optional[matplotlib.figure.Figure] = None,
        ax: Optional[matplotlib.axes.Axes] = None,
        **kwargs: Any,
    ):
        """
        Initialize Radar Plot.

        Parameters
        ----------
        data : Any, optional
            Input data containing observations and model predictions.
        obs_col : str, optional
            Column/variable name for observations.
        mod_cols : str or list of str, optional
            Column/variable names for model predictions.
        metrics : list of str, optional
            List of metrics to calculate. Defaults to ['R', 'NMB', 'NME', 'RMSE', 'MAE'].
        metrics_data : xr.Dataset, optional
            Pre-calculated normalized metrics. Should have metrics as variables
            and models as a dimension (e.g., 'model').
        fig : matplotlib.figure.Figure, optional
            An existing Figure object.
        ax : matplotlib.axes.Axes, optional
            An existing polar Axes object.
        **kwargs : Any
            Arguments passed to BasePlot. 'subplot_kw={"projection": "polar"}'
            is added automatically if ax is None.
        """
        if ax is None and "subplot_kw" not in kwargs:
            kwargs["subplot_kw"] = {"projection": "polar"}

        super().__init__(fig=fig, ax=ax, **kwargs)

        if metrics_data is not None:
            self.metrics_data = metrics_data
        elif data is not None and obs_col is not None and mod_cols is not None:
            self._calculate_all_metrics(data, obs_col, mod_cols, metrics)
        else:
            raise ValueError(
                "Must provide either metrics_data or data with obs_col and mod_cols"
            )

        # Ensure we have a 'model' dimension if it's a single model without one
        if "model" not in self.metrics_data.dims:
            self.metrics_data = self.metrics_data.expand_dims("model")

    def _calculate_all_metrics(
        self,
        data: Any,
        obs_col: str,
        mod_cols: Union[str, List[str]],
        metrics: Optional[List[str]],
    ) -> None:
        """Calculate normalized metrics for one or more models."""
        if isinstance(mod_cols, str):
            mod_cols = [mod_cols]

        data_xr = normalize_data(data, prefer_xarray=True)
        obs = data_xr[obs_col]

        model_results = []
        for mod_col in mod_cols:
            mod = data_xr[mod_col]
            # Use all non-specified dimensions for reduction by default
            # In most cases for radar plots, we want a single value per model/metric
            ds = compute_radar_metrics(obs, mod, metrics=metrics)
            ds = ds.assign_coords(model=mod_col)
            model_results.append(ds)

        self.metrics_data = xr.concat(model_results, dim="model")
        _update_history(self.metrics_data, "Calculated metrics for RadarPlot")

    def plot(self, **kwargs: Any) -> matplotlib.axes.Axes:
        """Generate the radar chart.

        Parameters
        ----------
        **kwargs : Any
            Keyword arguments passed to `ax.plot` and `ax.fill`.

        Returns
        -------
        matplotlib.axes.Axes
            The polar axes object with the radar chart.
        """
        # Variables to plot
        variables = list(self.metrics_data.data_vars)
        num_vars = len(variables)

        # Compute angle for each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

        # The plot is circular, so we need to "complete the loop"
        # and append the start value to the end.
        angles += angles[:1]

        # Draw one axe per variable + add labels
        self.ax.set_xticks(angles[:-1])
        self.ax.set_xticklabels(variables)

        # Draw ylabels
        self.ax.set_rlabel_position(0)
        self.ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        self.ax.set_yticklabels(
            ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=7
        )
        self.ax.set_ylim(0, 1)

        # Plot each model
        for model_name in self.metrics_data.model.values:
            model_ds = self.metrics_data.sel(model=model_name)
            values = [float(model_ds[var].values) for var in variables]
            values += values[:1]

            line_kwargs = {
                "linewidth": 2,
                "linestyle": "solid",
                "label": str(model_name),
            }
            line_kwargs.update(kwargs)

            self.ax.plot(angles, values, **line_kwargs)
            self.ax.fill(angles, values, alpha=0.1)

        self.ax.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))

        return self.ax

    def hvplot(self, **kwargs: Any) -> Any:
        """Generate an interactive radar chart using hvPlot (Track B).

        Note: hvPlot/HoloViews doesn't have a native "radar" plot type that
        works exactly like Matplotlib's polar projection for this purpose,
        but we can represent it as a polar line plot.

        Parameters
        ----------
        **kwargs : Any
            Keyword arguments passed to plotting function.

        Returns
        -------
        holoviews.core.Element
            The interactive radar chart.
        """
        try:
            import hvplot.xarray  # noqa: F401
        except ImportError:
            raise ImportError(
                "hvplot and holoviews are required for interactive plotting."
            )

        # Prepare data for hvplot
        variables = list(self.metrics_data.data_vars)

        # Melt the dataset to have a 'metric' dimension
        ds_melted = self.metrics_data.to_array(dim="metric", name="value")

        # Add angles for polar plot
        num_vars = len(variables)
        angles = np.linspace(0, 360, num_vars, endpoint=False)
        angle_da = xr.DataArray(angles, coords={"metric": variables}, dims=["metric"])
        ds_melted = ds_melted.assign_coords(angle=angle_da)

        # Close the loop for each model
        # This is a bit tricky with hvplot/xarray natively without manual manipulation
        # For simplicity, we'll return a polar scatter/line plot

        return ds_melted.hvplot.line(
            x="angle",
            y="value",
            by="model",
            polar=True,
            ylim=(0, 1),
            hover_cols=["metric"],
            **kwargs,
        )
