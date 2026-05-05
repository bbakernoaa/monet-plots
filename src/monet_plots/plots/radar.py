from typing import Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ..plot_utils import _update_history, normalize_data
from ..verification_metrics import compute_radar_metrics
from .base import BasePlot


class RadarPlot(BasePlot):
    """
    Radar (Spider) Chart for multi-model performance evaluation.

    Normalizes various performance metrics (Correlation, NMB, NME, RMSE, MAE)
    to a 0-1 scale where 1 is perfect performance.

    Attributes
    ----------
    metrics_data : xr.Dataset
        The normalized metrics for each model.
    """

    def __init__(
        self,
        data: Any,
        obs_col: str = "obs",
        mod_cols: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        fig: Optional[plt.Figure] = None,
        ax: Optional[plt.Axes] = None,
        **kwargs: Any,
    ):
        """
        Initialize the RadarPlot.

        Parameters
        ----------
        data : Any
            Input data (DataFrame, Dataset, or DataArray).
        obs_col : str, default "obs"
            Name of the observation column/variable.
        mod_cols : List[str], optional
            Names of the model columns/variables to evaluate.
            If None, all variables except obs_col are used.
        metrics : List[str], optional
            List of metrics to compute. Defaults to ['R', 'NMB', 'NME', 'RMSE', 'MAE'].
        fig : plt.Figure, optional
            Matplotlib figure.
        ax : plt.Axes, optional
            Matplotlib axes.
        **kwargs : Any
            Passed to BasePlot.

        Examples
        --------
        >>> ds = xr.Dataset({"obs": (["index"], [1, 2]), "mod": (["index"], [1.1, 1.9])})
        >>> plot = RadarPlot(ds, obs_col="obs")
        """
        # Ensure polar projection for radar charts
        subplot_kw = kwargs.pop("subplot_kw", {})
        subplot_kw.setdefault("projection", "polar")

        super().__init__(fig=fig, ax=ax, subplot_kw=subplot_kw, **kwargs)

        data_xr = normalize_data(data, prefer_xarray=True)
        if isinstance(data_xr, xr.DataArray):
            data_xr = data_xr.to_dataset()

        if mod_cols is None:
            mod_cols = [v for v in data_xr.data_vars if v != obs_col]

        if metrics is None:
            metrics = ["R", "NMB", "NME", "RMSE", "MAE"]

        obs = data_xr[obs_col]

        model_results = []
        for mod_col in mod_cols:
            mod = data_xr[mod_col]
            ds = compute_radar_metrics(obs, mod, metrics=metrics)
            ds = ds.assign_coords(model=mod_col)
            model_results.append(ds)

        self.metrics_data = xr.concat(model_results, dim="model")
        _update_history(self.metrics_data, "Calculated metrics for RadarPlot")

    def plot(self, **kwargs: Any) -> plt.Axes:
        """
        Generate the radar chart (Track A).

        Parameters
        ----------
        **kwargs : Any
            Passed to ax.plot.

        Returns
        -------
        plt.Axes
            The Matplotlib axes.

        Examples
        --------
        >>> ax = plot.plot(color='blue')
        """
        metrics = list(self.metrics_data.data_vars)
        num_vars = len(metrics)

        # Compute angles for each metric
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Close the loop

        self.ax.set_theta_offset(np.pi / 2)
        self.ax.set_theta_direction(-1)

        # Draw labels
        self.ax.set_xticks(angles[:-1])
        self.ax.set_xticklabels(metrics)

        # Plot each model
        for model_name in self.metrics_data.model.values:
            # Vectorized extraction of values for this model
            values = self.metrics_data.sel(model=model_name).to_array().values.tolist()
            values += values[:1]  # Close the loop

            line_kwargs = {
                "linewidth": 2,
                "linestyle": "solid",
                "label": str(model_name),
            }
            line_kwargs.update(kwargs)

            (line,) = self.ax.plot(angles, values, **line_kwargs)
            color = line.get_color()
            self.ax.fill(angles, values, color=color, alpha=0.1)

        self.ax.set_ylim(0, 1)
        self.ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

        return self.ax

    def hvplot(self, **kwargs: Any) -> Any:
        """
        Generate an interactive radar chart (Track B).

        Parameters
        ----------
        **kwargs : Any
            Passed to hvplot.

        Returns
        -------
        Any
            The HoloViews object.

        Examples
        --------
        >>> hv_obj = plot.hvplot()
        """
        import hvplot.xarray  # noqa: F401

        # Track B: Use a polar line plot if supported, otherwise line
        # hvPlot doesn't have a direct 'radar' but we can use polar projection in some cases
        # For now, keeping it as a multi-line plot for clear multi-model comparison
        return self.metrics_data.to_array(dim="metric").hvplot.line(
            x="metric", by="model", **kwargs
        )
