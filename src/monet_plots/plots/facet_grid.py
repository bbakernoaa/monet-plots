# src/monet_plots/plots/facet_grid.py
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xarray as xr

from ..plot_utils import to_dataframe
from ..style import wiley_style
from .base import BasePlot

if TYPE_CHECKING:
    import cartopy.crs as ccrs


class FacetGridPlot(BasePlot):
    """Creates a facet grid plot.

    This class creates a facet grid plot using seaborn's FacetGrid.
    """

    def __init__(
        self,
        data: Any,
        row: str | None = None,
        col: str | None = None,
        hue: str | None = None,
        col_wrap: int | None = None,
        height: float = 3,
        aspect: float = 1,
        subplot_kws: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initializes the facet grid.

        Args:
            data: The data to plot.
            row (str, optional): Variable to map to row facets. Defaults to None
            col (str, optional): Variable to map to column facets. Defaults to None
            hue (str, optional): Variable to map to color mapping. Defaults to None
            col_wrap (int, optional): Number of columns before wrapping. Defaults to None
            height (float, optional): Height of each facet in inches. Defaults to 3
            aspect (float, optional): Aspect ratio of each facet. Defaults to 1
            subplot_kws (dict, optional): Keyword arguments for subplots (e.g. projection).
            **kwargs: Additional keyword arguments to pass to `FacetGrid`.
        """
        # Apply Wiley style
        plt.style.use(wiley_style)

        # Store facet parameters
        self.row = row
        self.col = col
        self.hue = hue
        self.col_wrap = col_wrap
        self.height = height
        self.aspect = aspect

        # Convert data to pandas DataFrame and ensure coordinates are columns
        self.raw_data = data
        self.data = to_dataframe(data).reset_index()

        # Create the FacetGrid (this creates its own figure)
        self.grid = sns.FacetGrid(
            self.data,
            row=self.row,
            col=self.col,
            hue=self.hue,
            col_wrap=self.col_wrap,
            height=self.height,
            aspect=self.aspect,
            subplot_kws=subplot_kws,
            **kwargs,
        )

        # Initialize BasePlot with the figure and first axes from the grid
        axes = self.grid.axes.flatten()
        super().__init__(fig=self.grid.fig, ax=axes[0])

        # For compatibility with tests, also store as 'g'
        self.g = self.grid

    def map_dataframe(self, plot_func: Callable, *args: Any, **kwargs: Any) -> None:
        """Maps a plotting function to the facet grid.

        Args:
            plot_func (function): The plotting function to map.
            *args: Positional arguments to pass to the plotting function.
            **kwargs: Keyword arguments to pass to the plotting function.
        """
        self.grid.map_dataframe(plot_func, *args, **kwargs)

    def set_titles(self, *args, **kwargs):
        """Sets the titles of the facet grid.

        Args:
            *args: Positional arguments to pass to `set_titles`.
            **kwargs: Keyword arguments to pass to `set_titles`.
        """
        self.grid.set_titles(*args, **kwargs)

    def save(self, filename, **kwargs):
        """Saves the plot to a file.

        Args:
            filename (str): The name of the file to save the plot to.
            **kwargs: Additional keyword arguments to pass to `savefig`.
        """
        self.fig.savefig(filename, **kwargs)

    def plot(self, plot_func=None, *args, **kwargs):
        """Plots the data using the FacetGrid.

        Args:
            plot_func (function, optional): The plotting function to use.
            *args: Positional arguments to pass to the plotting function.
            **kwargs: Keyword arguments to pass to the plotting function.
        """
        if plot_func is not None:
            self.grid.map(plot_func, *args, **kwargs)

    def close(self):
        """Closes the plot."""
        plt.close(self.fig)


class SpatialFacetGridPlot(FacetGridPlot):
    """Geospatial version of FacetGridPlot."""

    def __init__(
        self,
        data: xr.DataArray | xr.Dataset,
        *,
        row: str | None = None,
        col: str | None = None,
        col_wrap: int | None = None,
        projection: ccrs.Projection | None = None,
        height: float = 4,
        aspect: float = 1.2,
        **kwargs: Any,
    ) -> None:
        """Initialize Spatial Facet Grid.

        Parameters
        ----------
        data : xr.DataArray or xr.Dataset
            Geospatial data to plot.
        row : str, optional
            Dimension/variable to map to rows.
        col : str, optional
            Dimension/variable to map to columns.
        col_wrap : int, optional
            Wrap columns at this number.
        projection : ccrs.Projection, optional
            Cartopy projection for the maps. Defaults to PlateCarree.
        height : float
            Height of each facet.
        aspect : float
            Aspect ratio of each facet.
        **kwargs : Any
            Additional arguments for FacetGrid.
        """
        self.original_data = data
        import cartopy.crs as ccrs

        self.projection = projection or ccrs.PlateCarree()

        # Handle xr.Dataset by converting to DataArray if faceting by variable
        self.is_dataset = isinstance(data, xr.Dataset)
        if self.is_dataset:
            if row == "variable" or col == "variable":
                data = data.to_array(dim="variable", name="value")

        super().__init__(
            data,
            row=row,
            col=col,
            col_wrap=col_wrap,
            height=height,
            aspect=aspect,
            subplot_kws={"projection": self.projection},
            **kwargs,
        )

        # Set default titles
        self._set_default_titles()

    def _set_default_titles(self) -> None:
        """Format facet titles with metadata and date-time."""
        for ax in self.grid.axes.flatten():
            if ax is None:
                continue
            title = ax.get_title()
            if " = " in title:
                dim, val = title.split(" = ")
                dim = dim.strip()
                val = val.strip()

                # Handle date-time formatting
                try:
                    dt = pd.to_datetime(val)
                    val = dt.strftime("%Y-%m-%d %H:%M")
                except (ValueError, TypeError):
                    pass

                # Handle long_name for dimensions/variables
                if dim == "variable" and self.is_dataset:
                    try:
                        # self.original_data is the original Dataset
                        var_obj = self.original_data[val]
                        long_name = var_obj.attrs.get("long_name", val)
                        units = var_obj.attrs.get("units", "")
                        val = f"{long_name} ({units})" if units else long_name
                        dim = ""
                    except (KeyError, AttributeError):
                        pass
                elif dim in self.original_data.coords:
                    try:
                        coord_obj = self.original_data.coords[dim]
                        long_name = coord_obj.attrs.get("long_name", dim)
                        dim = long_name
                    except (KeyError, AttributeError):
                        pass

                ax.set_title(f"{dim} {val}".strip())

    def add_map_features(self, **kwargs: Any) -> None:
        """Add cartopy features to all facets.

        Parameters
        ----------
        **kwargs : Any
            Keyword arguments passed to SpatialPlot.add_features.
            Default is coastlines=True.
        """
        from .spatial import SpatialPlot

        if "coastlines" not in kwargs:
            kwargs["coastlines"] = True

        for ax in self.grid.axes.flatten():
            if ax is None:
                continue
            # Use SpatialPlot's feature logic on each axis
            SpatialPlot(ax=ax, projection=self.projection, **kwargs)

    def map_monet(
        self,
        plotter_class: type,
        *,
        x: str = "lon",
        y: str = "lat",
        var_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Map a monet-plots spatial plotter to the grid.

        Parameters
        ----------
        plotter_class : type
            A class from monet_plots.plots (e.g., SpatialImshowPlot).
        x : str
            Column name for longitude.
        y : str
            Column name for latitude.
        var_name : str, optional
            The variable name to plot. If None and faceting by variable,
            uses 'value'.
        **kwargs : Any
            Arguments passed to the plotter and map features.
        """
        if var_name is None:
            if "variable" in self.data.columns:
                var_name = "value"
            elif isinstance(self.raw_data, xr.DataArray):
                var_name = self.raw_data.name
            elif isinstance(self.raw_data, xr.Dataset):
                # If not faceting by variable, we need a var_name
                # For now just pick the first data var if not provided
                var_name = list(self.raw_data.data_vars)[0]

        def _mapped_plot(*args, **kwargs_inner):
            data_df = kwargs_inner.pop("data")
            ax = plt.gca()

            # Reconstruct DataArray from DataFrame
            # We assume x and y are the coordinates
            temp_da = data_df.set_index([y, x]).to_xarray()[var_name]

            # Create plotter instance
            plotter = plotter_class(temp_da, ax=ax, **kwargs_inner)
            plotter.plot()

        # Separate feature kwargs
        feature_keys = [
            "coastlines",
            "states",
            "countries",
            "ocean",
            "land",
            "lakes",
            "rivers",
            "borders",
            "gridlines",
            "extent",
            "resolution",
        ]
        feature_kwargs = {k: kwargs.pop(k) for k in feature_keys if k in kwargs}

        # Check for colorbar requirement before popping from kwargs
        add_shared_cb = kwargs.pop("add_colorbar", False)

        self.map_dataframe(_mapped_plot, **kwargs)

        # Add features
        self.add_map_features(**feature_kwargs)

        # Add shared colorbar if requested
        if add_shared_cb:
            self._add_shared_colorbar(**kwargs)

    def _add_shared_colorbar(self, **kwargs: Any) -> None:
        """Add a shared colorbar to the figure."""
        # Find the last mappable object in the facets and the last valid axis
        mappable = None
        target_ax = None
        for ax in reversed(self.grid.axes.flatten()):
            if ax is None:
                continue
            if target_ax is None:
                target_ax = ax
            if ax.collections and mappable is None:
                mappable = ax.collections[0]
            if ax.images and mappable is None:
                mappable = ax.images[0]

        if mappable and target_ax:
            # Add colorbar to the last valid axis
            self.add_colorbar(
                mappable,
                ax=target_ax,
                label=kwargs.get("label", ""),
            )
