from __future__ import annotations

from typing import Any

import matplotlib
import pandas as pd
import seaborn as sns
import xarray as xr

from .base import BasePlot


class GroupedDistributionPlot(BasePlot):
    """
    Generates a grouped boxplot comparison from multiple pre-processed xarray
    DataArrays or a Dataset, organized by a grouping dimension.

    This plot follows the Aero Protocol, supporting lazy evaluation of Xarray/Dask
    objects and providing both static (Matplotlib) and interactive (hvPlot)
    visualizations.
    """

    def __init__(
        self,
        data: xr.DataArray | xr.Dataset | pd.DataFrame | list[xr.DataArray],
        *,
        labels: list[str] | None = None,
        group_dim: str = "region",
        var_label: str = "Value",
        hue: str = "Model",
        **kwargs: Any,
    ):
        """
        Initialize the GroupedDistributionPlot.

        Parameters
        ----------
        data : xr.DataArray | xr.Dataset | pd.DataFrame | list[xr.DataArray]
            Input data. If a list of DataArrays, they will be combined.
        labels : list[str], optional
            Names for the legend. Defaults to None.
        group_dim : str, optional
            The dimension name representing the groups/categories.
            Defaults to "region".
        var_label : str, optional
            Label for the Y-axis. Defaults to "Value".
        hue : str, optional
            Column name for the hue (legend). Defaults to "Model".
        **kwargs : Any
            Additional keyword arguments passed to BasePlot.
        """
        super().__init__(**kwargs)
        self.data = data
        self.labels = labels
        self.group_dim = group_dim
        self.var_label = var_label
        self.hue = hue
        self.df_plot = None
        self._update_provenance()

    def _update_provenance(self) -> None:
        """Update the history attribute for provenance tracking."""

        def _update_obj(obj):
            if hasattr(obj, "attrs"):
                history = obj.attrs.get("history", "")
                obj.attrs["history"] = (
                    f"Plotted with monet-plots.GroupedDistributionPlot; {history}"
                )

        if isinstance(self.data, list):
            for item in self.data:
                _update_obj(item)
        else:
            _update_obj(self.data)

    def _prepare_data(self) -> None:
        """Prepare data for plotting by converting it to a long-format DataFrame."""
        if self.df_plot is not None:
            return

        dfs = []

        # Standardize labels (hue labels)
        labels = self.labels
        if labels is None:
            if isinstance(self.data, list):
                labels = [
                    da.name if hasattr(da, "name") and da.name else f"Model {i + 1}"
                    for i, da in enumerate(self.data)
                ]
            elif isinstance(self.data, xr.Dataset):
                labels = list(self.data.data_vars)
            elif isinstance(self.data, xr.DataArray):
                labels = [self.data.name if self.data.name else "Model"]
            else:
                labels = ["Model"]
        self.labels = labels

        # Try to determine var_label from data if it's default
        if self.var_label == "Value":
            if isinstance(self.data, list) and len(self.data) > 0:
                first_da = self.data[0]
                if hasattr(first_da, "name") and first_da.name:
                    self.var_label = first_da.name
            elif isinstance(self.data, xr.DataArray):
                if self.data.name:
                    self.var_label = self.data.name

        # Handle different input types
        if isinstance(self.data, list):
            for da, label in zip(self.data, labels):
                dfs.append(self._process_da(da, label))
        elif isinstance(self.data, xr.Dataset):
            for var in labels:
                dfs.append(self._process_da(self.data[var], var))
        elif isinstance(self.data, xr.DataArray):
            dfs.append(self._process_da(self.data, labels[0]))
        elif isinstance(self.data, pd.DataFrame):
            self.df_plot = self.data
            return

        if not dfs:
            raise ValueError("No data found to plot.")

        self.df_plot = pd.concat(dfs, ignore_index=True)

    def _process_da(self, da: xr.DataArray, label: str) -> pd.DataFrame:
        """Helper to process a single DataArray into a long-format DataFrame."""
        if self.group_dim not in da.dims:
            raise ValueError(
                f"Dimension '{self.group_dim}' not found in DataArray for {label}"
            )

        # Convert to series to avoid issues with non-dimension coordinates
        # being included as columns during to_dataframe() or reset_index failures
        # when a dimension name overlaps with another coordinate name.
        # to_series() handles all dimensions and results in a MultiIndex series.
        df_temp = da.to_series().to_frame(name="value").reset_index()
        df_temp = df_temp[[self.group_dim, "value"]]
        df_temp[self.hue] = label
        return df_temp.dropna(subset=["value"])

    def plot(self, **kwargs: Any) -> matplotlib.axes.Axes:
        """
        Generate the grouped distribution plot.

        Parameters
        ----------
        **kwargs : Any
            Additional keyword arguments passed to sns.boxplot.

        Returns
        -------
        matplotlib.axes.Axes
            The matplotlib axes containing the plot.
        """
        from ..style import CB_COLORS

        self._prepare_data()

        # Default styling
        boxplot_kwargs = {
            "x": self.group_dim,
            "y": "value",
            "hue": self.hue,
            "data": self.df_plot,
            "showfliers": False,
            "width": 0.7,
            "linewidth": 1.2,
        }

        # Default palette
        if "palette" not in kwargs:
            unique_hues = self.df_plot[self.hue].unique()
            if len(unique_hues) == 2:
                # Use blue and orange to match the reference image
                boxplot_kwargs["palette"] = {
                    unique_hues[0]: CB_COLORS[5],
                    unique_hues[1]: CB_COLORS[1],
                }
            else:
                boxplot_kwargs["palette"] = CB_COLORS

        boxplot_kwargs.update(kwargs)

        sns.boxplot(ax=self.ax, **boxplot_kwargs)

        # Formatting
        self.ax.set_ylabel(self.var_label, fontweight="bold")
        self.ax.set_xlabel(self.group_dim.capitalize(), fontweight="bold")

        # Handle title
        title = f"{self.group_dim.capitalize()} Distribution of {self.var_label}"
        labels = self.df_plot[self.hue].unique().tolist()
        if len(labels) >= 2:
            if len(labels) == 2:
                title += f": {labels[0]} vs {labels[1]}"
            else:
                title += f": {', '.join(labels[:-1])} and {labels[-1]}"

        self.ax.set_title(title, fontsize=12, fontweight="bold", pad=15)

        # Legend
        self.ax.legend(title=self.hue.lower(), loc="upper right")

        return self.ax

    def hvplot(self, **kwargs: Any) -> Any:
        """
        Generate an interactive grouped distribution plot using hvPlot.

        Parameters
        ----------
        **kwargs : Any
            Additional keyword arguments passed to hvplot.box.

        Returns
        -------
        holoviews.core.layout.Layout
            The interactive hvPlot object.
        """
        import hvplot.pandas  # noqa: F401

        self._prepare_data()

        plot_kwargs = {
            "by": self.group_dim,
            "y": "value",
            "c": self.hue,
            "kind": "box",
            "title": f"Distribution of {self.var_label} by {self.group_dim}",
        }
        plot_kwargs.update(kwargs)

        return self.df_plot.hvplot(**plot_kwargs)
