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

    def __init__(
        self,
        data: Any,
        row: str = None,
        col: str = None,
        hue: str = None,
        col_wrap: int = None,
        height: float = 3,
        aspect: float = 1,
        **kwargs,
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
            plot_func (function, optional): The plotting function to use.
            *args: Positional arguments to pass to the plotting function.
            **kwargs: Keyword arguments to pass to the plotting function.
        """
        if plot_func is not None:
            self.grid.map(plot_func, *args, **kwargs)

    def close(self):
        """Closes the plot."""
        plt.close(self.fig)
