# src/monet_plots/plots/scatter.py

import seaborn as sns
import numpy as np
from .base import BasePlot
from ..plot_utils import to_dataframe
from typing import Any, Union, List


class ScatterPlot(BasePlot):
    """Create a scatter plot with a regression line.

    This plot shows the relationship between two variables and includes a
    linear regression model fit.
    """

    def __init__(
        self,
        df: Any,
        x: str,
        y: Union[str, List[str]],
        c: str = None,
        colorbar: bool = False,
        title: str = None,
        *args,
        **kwargs,
    ):
        """
        Initialize the plot with data and plot settings.

        Args:
            df (pd.DataFrame, np.ndarray, xr.Dataset, xr.DataArray): DataFrame with the data to plot.
            x (str): Column name for the x-axis.
            y (str or list): Column name(s) for the y-axis.
            c (str, optional): Column name for colorizing the points.
            colorbar (bool): Whether to add a colorbar.
            title (str, optional): Title for the plot.
        """
        super().__init__(*args, **kwargs)
        if self.ax is None:
            self.ax = self.fig.add_subplot(1, 1, 1)
        self.x = x
        if isinstance(y, str):
            self.y = [y]
        else:
            self.y = y
        self.c = c
        self.colorbar = colorbar

        required_cols = [self.x] + self.y
        if self.c is not None:
            required_cols.append(self.c)
        self.df = to_dataframe(df).dropna(subset=required_cols)
        self.title = title

    def plot(self, **kwargs):
        """Generate the scatter plot."""
        for y_col in self.y:
            if self.c is not None:
                # Use plt.scatter for more control over color mapping
                mappable = self.ax.scatter(
                    self.df[self.x], self.df[y_col], c=self.df[self.c], **kwargs
                )
                if self.colorbar:
                    self.add_colorbar(mappable)
                # Add regression line manually
                m, b = np.polyfit(self.df[self.x], self.df[y_col], 1)
                self.ax.plot(self.df[self.x], m * self.df[self.x] + b, color="red")
            else:
                sns.regplot(
                    data=self.df, x=self.x, y=y_col, label=y_col, ax=self.ax, **kwargs
                )

        if len(self.y) > 1 and self.c is None:
            self.ax.legend()
        if self.title:
            self.ax.set_title(self.title)
        return self.ax

    def hvplot(self, **kwargs):
        """Generate an interactive scatter plot using hvPlot."""
        import hvplot.pandas  # noqa: F401

        plot_kwargs = {"x": self.x, "y": self.y}
        if self.c:
            plot_kwargs["c"] = self.c
        if self.title:
            plot_kwargs["title"] = self.title

        plot_kwargs.update(kwargs)
        return self.df.hvplot.scatter(**plot_kwargs)
