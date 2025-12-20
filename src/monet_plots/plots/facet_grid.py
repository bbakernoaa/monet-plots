# src/monet_plots/plots/facet_grid.py
from .base import BasePlot
import seaborn as sns
import matplotlib.pyplot as plt
from ..style import wiley_style
from ..plot_utils import to_dataframe
from typing import Any


class FacetGridPlot(BasePlot):
    """Creates a facet grid plot.

    This class creates a facet grid plot using seaborn's FacetGrid.
    """

    def __init__(self, data: Any, row: str = None, col: str = None, hue: str = None, col_wrap: int = None,
                 height: float = 3, aspect: float = 1, **kwargs):
        """Initializes the facet grid.

        Args:
            data (pd.DataFrame, np.ndarray, xr.Dataset, xr.DataArray): The data to plot.
            row (str, optional): Variable to map to row facets. Defaults to None
            col (str, optional): Variable to map to column facets. Defaults to None
            hue (str, optional): Variable to map to color mapping. Defaults to None
            col_wrap (int, optional): Number of columns before wrapping. Defaults to None
            height (float, optional): Height of each facet in inches. Defaults to 3
            aspect (float, optional): Aspect ratio of each facet. Defaults to 1
            **kwargs: Additional keyword arguments to pass to `FacetGrid`.
        """
        # Apply Wiley style - this is the key functionality from BasePlot
        plt.style.use(wiley_style)

        # Initialize BasePlot with general figure parameters
        super().__init__(**kwargs)

        # Convert data to pandas DataFrame
        data = to_dataframe(data)

        # Store the data and facet parameters
        self.data = data
        self.row = row
        self.col = col
        self.hue = hue
        self.col_wrap = col_wrap
        self.height = height
        self.aspect = aspect

        # Create the FacetGrid (this creates its own figure)
        self.grid = sns.FacetGrid(
            data,
            row=self.row,
            col=self.col,
            hue=self.hue,
            col_wrap=self.col_wrap,
            height=self.height,
            aspect=self.aspect,
            **kwargs
        )

        # Update the figure reference to the one from the grid since seaborn creates its own
        self.fig = self.grid.fig
        self.ax = None  # FacetGrid handles multiple axes internally

        # For compatibility with tests, also store as 'g'
        self.g = self.grid

    def map_dataframe(self, plot_func, *args, **kwargs):
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
            plot_func (function, optional): The plotting function to use. If None, uses the default plotting behavior.
            *args: Positional arguments to pass to the plotting function.
            **kwargs: Keyword arguments to pass to the plotting function.
        """
        if plot_func is not None:
            # Map the provided plotting function to the grid
            self.grid.map(plot_func, *args, **kwargs)
        else:
            # Default behavior: map a simple plot function if no specific function is provided
            # For xarray DataArrays, we can use the default plot method
            # The default is to call the default plot method on the data
            # This is typically used after FacetGrid is set up
            pass  # The FacetGrid is already created, user would typically call map after this

    def close(self):
        """Closes the plot."""
        plt.close(self.fig)
