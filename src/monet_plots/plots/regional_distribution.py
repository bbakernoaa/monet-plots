from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xarray as xr

from .base import BasePlot


class RegionalDistributionPlot(BasePlot):
    """
    Generates a grouped boxplot comparison from two pre-processed xarray DataArrays
    or a Dataset, organized by a grouping dimension (e.g., region).

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
        Initialize the RegionalDistributionPlot.

        Parameters
        ----------
        data : xr.DataArray | xr.Dataset | pd.DataFrame | list[xr.DataArray]
            Input data. If a list of DataArrays, they will be combined.
        labels : list[str], optional
            Names for the legend. Defaults to None.
        group_dim : str, optional
            The dimension name representing the regions. Defaults to "region".
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
                    f"Plotted with monet-plots.RegionalDistributionPlot; {history}"
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

        # Standardize labels
        labels = self.labels
        if labels is None:
            if isinstance(self.data, list):
                labels = [f"Model {i + 1}" for i in range(len(self.data))]
            elif isinstance(self.data, xr.Dataset):
                labels = list(self.data.data_vars)
            elif isinstance(self.data, xr.DataArray):
                labels = [self.data.name if self.data.name else "Model"]
            else:
                labels = ["Model"]

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

        non_group_dims = [d for d in da.dims if d != self.group_dim]

        if non_group_dims:
            da_stacked = da.stack(samples=non_group_dims)
        else:
            da_stacked = da

        # to_dataframe() will compute dask arrays
        df_temp = da_stacked.to_dataframe(name="value").reset_index()
        df_temp = df_temp[[self.group_dim, "value"]]
        df_temp[self.hue] = label
        return df_temp.dropna(subset=["value"])

    def plot(self, **kwargs: Any) -> plt.Axes:
        """
        Generate the regional distribution plot.

        Parameters
        ----------
        **kwargs : Any
            Additional keyword arguments passed to sns.boxplot.

        Returns
        -------
        plt.Axes
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
        self.ax.set_xlabel("")

        # Handle title if labels are available
        labels = self.labels
        if labels is None:
            labels = self.df_plot[self.hue].unique().tolist()

        if len(labels) >= 2:
            if len(labels) == 2:
                title = f"Regional comparisons of {labels[0]} to {labels[1]} Total {self.var_label}"
            else:
                title = f"Regional comparisons of {', '.join(labels[:-1])} and {labels[-1]} Total {self.var_label}"
            self.ax.set_title(title, fontsize=12, fontweight="bold", pad=15)

        # Legend
        self.ax.legend(title=self.hue.lower(), loc="upper right")

        return self.ax

    def add_inset_map(
        self, extent: list[float] | None = None, **kwargs: Any
    ) -> plt.Axes:
        """
        Add a geospatial inset map to the plot.

        Parameters
        ----------
        extent : list[float], optional
            Geographic extent [lon_min, lon_max, lat_min, lat_max].
        **kwargs : Any
            Additional keyword arguments passed to SpatialPlot.
            Use 'inset_pos' to specify the position [left, bottom, width, height]
            in axes coordinates. Default is [0.2, 0.5, 0.3, 0.4].

        Returns
        -------
        plt.Axes
            The inset axes.
        """
        from .spatial import SpatialPlot
        import cartopy.crs as ccrs

        # Default position for the inset map
        inset_pos = kwargs.pop("inset_pos", [0.2, 0.5, 0.3, 0.4])

        # Create inset axes with the specified projection
        projection = kwargs.pop("projection", ccrs.PlateCarree())
        ax_inset = self.ax.inset_axes(inset_pos, projection=projection)

        # Ensure coastlines are enabled by default for the inset
        if "coastlines" not in kwargs:
            kwargs["coastlines"] = True

        # Use SpatialPlot to handle features and extent
        SpatialPlot(ax=ax_inset, projection=projection, extent=extent, **kwargs)

        return ax_inset

    def hvplot(self, **kwargs: Any) -> Any:
        """
        Generate an interactive regional distribution plot using hvPlot.

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
            "title": f"Regional Distribution of {self.var_label}",
        }
        plot_kwargs.update(kwargs)

        return self.df_plot.hvplot(**plot_kwargs)
