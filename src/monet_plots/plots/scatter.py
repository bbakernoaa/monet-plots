# src/monet_plots/plots/scatter.py

import matplotlib.pyplot as plt
import seaborn as sns
from .base import BasePlot
from ..plot_utils import to_dataframe
from typing import Any, Union, List

class ScatterPlot(BasePlot):
    """Create a scatter plot with a regression line.

    This plot shows the relationship between two variables and includes a
    linear regression model fit.
    """

    def __init__(self, df: Any, x: str, y: Union[str, List[str]], title: str = None, *args, **kwargs):
        """
        Initialize the plot with data and plot settings.

        Args:
            df (pd.DataFrame, np.ndarray, xr.Dataset, xr.DataArray): DataFrame with the data to plot.
            x (str): Column name for the x-axis.
            y (str or list): Column name(s) for the y-axis.
            title (str, optional): Title for the plot.
        """
        super().__init__(*args, **kwargs)
        self.x = x
        if isinstance(y, str):
            self.y = [y]
        else:
            self.y = y

        required_cols = [self.x] + self.y
        self.df = to_dataframe(df).dropna(subset=required_cols)
        self.title = title

    def plot(self, **kwargs):
        """Generate the scatter plot."""
        for y_col in self.y:
            sns.regplot(data=self.df, x=self.x, y=y_col, label=y_col, ax=self.ax, **kwargs)

        if len(self.y) > 1:
            self.ax.legend()
        if self.title:
            self.ax.set_title(self.title)
        return self.ax
