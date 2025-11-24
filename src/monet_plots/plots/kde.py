# src/monet_plots/plots/kde.py

import matplotlib.pyplot as plt
import seaborn as sns
from .base import BasePlot

class KDEPlot(BasePlot):
    """Create a kernel density estimate plot.

    This plot shows the distribution of a single variable.
    """

    def __init__(self, df, x, y, title=None, label=None, *args, **kwargs):
        """
        Initialize the plot with data and plot settings.

        Args:
            df (pd.DataFrame, np.ndarray, xr.Dataset, xr.DataArray): DataFrame with the data to plot.
            x (str): Column name for the x-axis.
            y (str): Column name for the y-axis.
            title (str, optional): Title for the plot.
            label (str, optional): Label for the plot.
        """
        super().__init__(*args, **kwargs)
        self.df = df
        self.x = x
        self.y = y
        self.title = title
        self.label = label

    def plot(self, **kwargs):
        """Generate the KDE plot."""
        with sns.axes_style("ticks"):
            self.ax = sns.kdeplot(data=self.df, x=self.x, y=self.y, ax=self.ax, label=self.label, **kwargs)
            if self.title:
                self.ax.set_title(self.title)
            sns.despine()
        return self.ax

