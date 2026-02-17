# src/monet_plots/plots/kde.py

import seaborn as sns
from .base import BasePlot


class KDEPlot(BasePlot):
    """Create a kernel density estimate plot.

    This plot shows the distribution of a single variable.
    """

    def __init__(
        self, df, x, y, title=None, label=None, *args, fig=None, ax=None, **kwargs
    ):
        """
        Initialize the plot with data and plot settings.

        Args:
            df (pd.DataFrame, np.ndarray, xr.Dataset, xr.DataArray): DataFrame with the data to plot.
            x (str): Column name for the x-axis.
            y (str): Column name for the y-axis.
            title (str, optional): Title for the plot.
            label (str, optional): Label for the plot.
        """
        super().__init__(fig=fig, ax=ax, **kwargs)
        self.df = df
        self.x = x
        self.y = y
        self.title = title
        self.label = label

    def plot(self, **kwargs):
        """Generate the KDE plot."""
        with sns.axes_style("ticks"):
            self.ax = sns.kdeplot(
                data=self.df, x=self.x, y=self.y, ax=self.ax, label=self.label, **kwargs
            )
            if self.title:
                self.ax.set_title(self.title)
            sns.despine()
        return self.ax

    def hvplot(self, **kwargs):
        """Generate an interactive KDE plot using hvPlot."""
        import hvplot.pandas  # noqa: F401
        import xarray as xr

        if self.y:
            # Bivariate KDE
            plot_kwargs = {"x": self.x, "y": self.y}
            if isinstance(self.df, (xr.DataArray, xr.Dataset)):
                import hvplot.xarray  # noqa: F401

                method = self.df.hvplot.bivariate
            else:
                method = self.df.hvplot.bivariate
        else:
            # Univariate KDE
            plot_kwargs = {"y": self.x}
            if isinstance(self.df, (xr.DataArray, xr.Dataset)):
                import hvplot.xarray  # noqa: F401

                method = self.df.hvplot.kde
            else:
                method = self.df.hvplot.kde

        if self.title:
            plot_kwargs["title"] = self.title
        if self.label:
            plot_kwargs["label"] = self.label

        plot_kwargs.update(kwargs)
        return method(**plot_kwargs)
