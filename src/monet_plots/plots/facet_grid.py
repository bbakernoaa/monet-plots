from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ..plot_utils import identify_coords, to_dataframe
from ..style import wiley_style
from .base import BasePlot

if TYPE_CHECKING:
    import xarray as xr


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
        **kwargs: Any,
    ):
        """Initializes the facet grid.

        Args:
            data: The data to plot.
            row (str, optional): Variable to map to row facets. Defaults to None
            col (str, optional): Variable to map to column facets. Defaults to None
            hue (str, optional): Variable to map to color mapping. Defaults to None
            col_wrap (int, optional): Number of columns before wrapping. Defaults to None
            height (float, optional): Height of each facet in inches. Defaults to 3
            aspect (float, optional): Aspect ratio of each facet. Defaults to 1
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
            **kwargs,
        )

        # Initialize BasePlot with the figure and first axes from the grid
        axes = self.grid.axes.flatten()
        super().__init__(fig=self.grid.fig, ax=axes[0])

        # For compatibility with tests, also store as 'g'
        self.g = self.grid

    def map_dataframe(self, plot_func: Any, *args: Any, **kwargs: Any) -> None:
        """Maps a plotting function to the facet grid.

        Args:
            plot_func (function): The plotting function to map.
            *args: Positional arguments to pass to the plotting function.
            **kwargs: Keyword arguments to pass to the plotting function.
        """
        self.grid.map_dataframe(plot_func, *args, **kwargs)

    def set_titles(self, *args: Any, **kwargs: Any) -> None:
        """Sets the titles of the facet grid.

        Args:
            *args: Positional arguments to pass to `set_titles`.
            **kwargs: Keyword arguments to pass to `set_titles`.
        """
        self.grid.set_titles(*args, **kwargs)

    def save(self, filename: str, **kwargs: Any) -> None:
        """Saves the plot to a file.

        Args:
            filename (str): The name of the file to save the plot to.
            **kwargs: Additional keyword arguments to pass to `savefig`.
        """
        self.fig.savefig(filename, **kwargs)

    def plot(self, plot_func: Any = None, *args: Any, **kwargs: Any) -> None:
        """Plots the data using the FacetGrid.

        Args:
            plot_func (function, optional): The plotting function to use.
            *args: Positional arguments to pass to the plotting function.
            **kwargs: Keyword arguments to pass to the plotting function.
        """
        if plot_func is not None:
            self.grid.map(plot_func, *args, **kwargs)

    def close(self) -> None:
        """Closes the plot."""
        plt.close(self.fig)


class SpatialFacetGridPlot(BasePlot):
    """Creates a geospatial facet grid plot using xarray.

    This class leverages xarray's FacetGrid to create multi-panel geospatial
    plots with cartopy projections and features.
    """

    def __init__(
        self,
        data: xr.DataArray | xr.Dataset,
        col: str | None = None,
        row: str | None = None,
        col_wrap: int | None = None,
        projection: ccrs.Projection = ccrs.PlateCarree(),
        subplot_kws: dict[str, Any] | None = None,
        plot_func: str = "imshow",
        **kwargs: Any,
    ):
        """Initializes the spatial facet grid.

        Parameters
        ----------
        data : xarray.DataArray or xarray.Dataset
            The data to plot.
        col : str, optional
            Dimension to facet along columns.
        row : str, optional
            Dimension to facet along rows.
        col_wrap : int, optional
            Wrap the column facet at this number.
        projection : ccrs.Projection, optional
            Cartopy projection for the facets, by default ccrs.PlateCarree().
        subplot_kws : dict, optional
            Additional keyword arguments for subplot creation.
        **kwargs : Any
            Additional keyword arguments passed to xarray's FacetGrid.
        """
        import xarray as xr

        plt.style.use(wiley_style)

        self.data = data
        self.col = col
        self.row = row
        self.col_wrap = col_wrap

        current_subplot_kws = subplot_kws.copy() if subplot_kws else {}
        current_subplot_kws.setdefault("projection", projection)

        # If data is a Dataset and no col/row provided, facet by variable by default
        if isinstance(data, xr.Dataset) and col is None and row is None:
            self.col = "variable"
            self.data = data.to_array()

        # Store unconsumed kwargs for the plot method
        # These might include map features like 'coastlines', 'states', etc.
        self.plot_kwargs = kwargs
        self.default_plot_func = plot_func

        # Initialize the xarray FacetGrid object
        from xarray.plot.facetgrid import FacetGrid

        self.grid = FacetGrid(
            self.data,
            col=self.col,
            row=self.row,
            col_wrap=self.col_wrap,
            subplot_kws=current_subplot_kws,
        )

        # Initialize BasePlot with the figure and first valid axes
        super().__init__(fig=self.grid.fig, ax=self.grid.axs.flatten()[0])

    @property
    def axs_flattened(self) -> np.ndarray:
        """Return a flattened array of all axes in the grid."""
        return self.grid.axs.flatten()

    def add_features(self, **kwargs: Any) -> None:
        """Add cartopy features to all facets in the grid.

        Parameters
        ----------
        **kwargs : Any
            Keyword arguments passed to SpatialPlot.add_features for each facet.
        """
        from .spatial import SpatialPlot

        for ax in self.axs_flattened:
            # We create a temporary SpatialPlot to use its add_features logic
            # This ensures consistent feature handling across all facets.
            SpatialPlot(ax=ax, **kwargs)

    def set_titles(self, template: str = "{kw} = {val}", **kwargs: Any) -> None:
        """Set titles for each facet.

        Parameters
        ----------
        template : str
            Title template.
        **kwargs : Any
            Additional arguments for set_titles.
        """
        self.grid.set_titles(template=template, **kwargs)

    def plot(self, plot_func: str | None = None, **kwargs: Any) -> Any:
        """Plot the data on the facet grid using xarray plotting.

        Parameters
        ----------
        plot_func : str, optional
            Xarray plotting method to use ('imshow', 'contourf', 'pcolormesh', 'contour').
            If None, uses the default_plot_func provided during initialization.
        **kwargs : Any
            Keyword arguments passed to the plotting method.

        Returns
        -------
        FacetGrid
            The xarray FacetGrid object.
        """
        if plot_func is None:
            plot_func = self.default_plot_func

        # Merge stored plot_kwargs with those passed to plot()
        final_kwargs = self.plot_kwargs.copy()
        final_kwargs.update(kwargs)

        # We need to map the appropriate xarray plotting function
        from xarray.plot import contour, contourf, imshow, pcolormesh

        mapping = {
            "imshow": imshow,
            "contourf": contourf,
            "contour": contour,
            "pcolormesh": pcolormesh,
        }

        func = mapping.get(plot_func, imshow)

        # Identify coordinates if not provided
        lon_coord, lat_coord = identify_coords(self.data)
        final_kwargs.setdefault("x", lon_coord)
        final_kwargs.setdefault("y", lat_coord)
        final_kwargs.setdefault("transform", ccrs.PlateCarree())

        # Extract map features before calling map_dataarray
        map_feature_keys = [
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
            "natural_earth",
        ]
        map_features = {
            k: final_kwargs.pop(k) for k in map_feature_keys if k in final_kwargs
        }

        x = final_kwargs.pop("x")
        y = final_kwargs.pop("y")

        # Map the plotting function
        self.grid.map_dataarray(func, x, y, **final_kwargs)

        # Add map features to all facets
        if map_features or "coastlines" not in map_features:
            # Default to coastlines if not specified otherwise
            if "coastlines" not in map_features:
                map_features["coastlines"] = True
            self.add_features(**map_features)

        return self.grid
