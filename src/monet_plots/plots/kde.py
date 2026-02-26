# src/monet_plots/plots/kde.py

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import seaborn as sns

from ..plot_utils import _update_history, normalize_data
from .base import BasePlot

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure


class KDEPlot(BasePlot):
    """Create a kernel density estimate plot.

    This plot shows the distribution of a single variable or the joint
    distribution of two variables. It supports lazy evaluation for large
    Xarray/Dask datasets by delaying computation until the plot call.

    Attributes
    ----------
    data : Union[xr.Dataset, xr.DataArray, pd.DataFrame]
        The input data for the plot.
    x : str
        The name of the variable for the x-axis.
    y : Optional[str]
        The name of the variable for the y-axis (for joint KDE).
    title : Optional[str]
        The title for the plot.
    label : Optional[str]
        The label for the plot.
    """

    def __init__(
        self,
        data: Any = None,
        x: Optional[str] = None,
        y: Optional[str] = None,
        title: Optional[str] = None,
        label: Optional[str] = None,
        fig: Optional[matplotlib.figure.Figure] = None,
        ax: Optional[matplotlib.axes.Axes] = None,
        df: Any = None,  # Backward compatibility alias
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize the KDE plot with data and plot settings.

        Parameters
        ----------
        data : Any, optional
            Input data. Can be a pandas DataFrame, xarray DataArray,
            xarray Dataset, or numpy ndarray, by default None.
        x : str, optional
            Variable name for the x-axis, by default None.
        y : str, optional
            Variable name for the y-axis (for bivariate KDE), by default None.
        title : str, optional
            Title for the plot, by default None.
        label : str, optional
            Label for the plot, by default None.
        fig : matplotlib.figure.Figure, optional
            An existing Figure object, by default None.
        ax : matplotlib.axes.Axes, optional
            An existing Axes object, by default None.
        df : Any, optional
            Alias for `data` for backward compatibility, by default None.
        *args : Any
            Additional positional arguments passed to BasePlot.
        **kwargs : Any
            Additional keyword arguments passed to BasePlot.
        """
        super().__init__(fig=fig, ax=ax, *args, **kwargs)
        self.data = normalize_data(data if data is not None else df)
        self.x = x
        self.y = y
        self.title = title
        self.label = label

        if not self.x:
            raise ValueError("Parameter 'x' must be provided.")

        _update_history(self.data, "Initialized KDEPlot")

    def plot(self, **kwargs: Any) -> matplotlib.axes.Axes:
        """Generate a static publication-quality KDE plot (Track A).

        Parameters
        ----------
        **kwargs : Any
            Additional keyword arguments passed to `sns.kdeplot`.

        Returns
        -------
        matplotlib.axes.Axes
            The axes object with the KDE plot.
        """
        # Performance: Compute required variables once to avoid double work
        cols = [self.x]
        if self.y:
            cols.append(self.y)

        if hasattr(self.data, "compute"):
            # Sub-selection before compute to minimize data transfer
            subset = self.data[cols]
            concrete_data = subset.compute()
        else:
            concrete_data = self.data

        with sns.axes_style("ticks"):
            self.ax = sns.kdeplot(
                data=concrete_data,
                x=self.x,
                y=self.y,
                ax=self.ax,
                label=self.label,
                **kwargs,
            )
            if self.title:
                self.ax.set_title(self.title)
            sns.despine()

        _update_history(self.data, "Generated KDEPlot")
        return self.ax

    def hvplot(self, **kwargs: Any) -> Any:
        """Generate an interactive KDE plot using hvPlot (Track B).

        Parameters
        ----------
        **kwargs : Any
            Keyword arguments passed to `hvplot.kde`.
            `rasterize=True` is recommended for high performance on large datasets.

        Returns
        -------
        holoviews.core.Element
            The interactive hvPlot object.
        """
        try:
            import hvplot.pandas  # noqa: F401
            import hvplot.xarray  # noqa: F401
        except ImportError:
            raise ImportError(
                "hvplot is required for interactive plotting. Install it with 'pip install hvplot'."
            )

        # Track B defaults
        plot_kwargs = {
            "x": self.x,
        }
        if self.y:
            plot_kwargs["y"] = self.y

        plot_kwargs.update(kwargs)

        return self.data.hvplot.kde(**plot_kwargs)
