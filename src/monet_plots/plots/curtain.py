# src/monet_plots/plots/curtain.py
"""Vertical curtain plot for cross-sectional data."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from ..plot_utils import _update_history, get_plot_kwargs, normalize_data
from .base import BasePlot

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure


class CurtainPlot(BasePlot):
    """Vertical curtain plot for cross-sectional data.

    This plot shows a 2D variable (e.g., concentration) as a function of
    one horizontal dimension (time or distance) and one vertical dimension
    (altitude or pressure). It supports lazy evaluation for large
    Xarray/Dask datasets by delaying computation until the plot call.

    Attributes
    ----------
    data : xr.DataArray
        The 2D input data for the plot.
    x : str
        Name of the x-axis dimension/coordinate (e.g., 'time').
    y : str
        Name of the y-axis dimension/coordinate (e.g., 'level').
    """

    def __init__(
        self,
        data: Any = None,
        *,
        x: Optional[str] = None,
        y: Optional[str] = None,
        fig: Optional[matplotlib.figure.Figure] = None,
        ax: Optional[matplotlib.axes.Axes] = None,
        df: Any = None,
        **kwargs: Any,
    ) -> None:
        """Initialize Curtain Plot.

        Parameters
        ----------
        data : Any, optional
            Input data. Should be a 2D xarray.DataArray or similar, by default None.
        x : str, optional
            Name of the x-axis dimension/coordinate (e.g., 'time'), by default None.
        y : str, optional
            Name of the y-axis dimension/coordinate (e.g., 'level'), by default None.
        fig : matplotlib.figure.Figure, optional
            An existing Figure object, by default None.
        ax : matplotlib.axes.Axes, optional
            An existing Axes object, by default None.
        df : Any, optional
            Alias for `data` for backward compatibility, by default None.
        **kwargs : Any
            Additional keyword arguments passed to BasePlot.
        """
        super().__init__(fig=fig, ax=ax, **kwargs)
        self.data = normalize_data(data if data is not None else df)
        self.x = x
        self.y = y

        _update_history(self.data, "Initialized CurtainPlot")

    def plot(
        self, kind: str = "pcolormesh", colorbar: bool = True, **kwargs: Any
    ) -> matplotlib.axes.Axes:
        """Generate a static publication-quality curtain plot (Track A).

        Parameters
        ----------
        kind : str, optional
            Type of plot ('pcolormesh' or 'contourf'), by default "pcolormesh".
        colorbar : bool, optional
            Whether to add a colorbar, by default True.
        **kwargs : Any
            Additional keyword arguments passed to the plotting function.

        Returns
        -------
        matplotlib.axes.Axes
            The axes object with the curtain plot.

        Raises
        ------
        ValueError
            If the data is not 2D or if `kind` is invalid.
        """
        plot_kwargs = get_plot_kwargs(**kwargs)

        # Performance: Compute required variables once to avoid double work
        if hasattr(self.data, "compute"):
            concrete_data = self.data.compute()
        else:
            concrete_data = self.data

        if concrete_data.ndim != 2:
            raise ValueError(
                f"CurtainPlot requires 2D data, got {concrete_data.ndim}D."
            )

        # Determine x and y if not provided
        if self.x is None:
            self.x = concrete_data.dims[1]
        if self.y is None:
            self.y = concrete_data.dims[0]

        if kind == "pcolormesh":
            mappable = self.ax.pcolormesh(
                concrete_data[self.x],
                concrete_data[self.y],
                concrete_data,
                shading="auto",
                **plot_kwargs,
            )
        elif kind == "contourf":
            mappable = self.ax.contourf(
                concrete_data[self.x],
                concrete_data[self.y],
                concrete_data,
                **plot_kwargs,
            )
        else:
            raise ValueError("kind must be 'pcolormesh' or 'contourf'")

        if colorbar:
            self.add_colorbar(mappable)

        self.ax.set_xlabel(self.x)
        self.ax.set_ylabel(self.y)

        _update_history(self.data, f"Generated CurtainPlot (kind={kind})")
        return self.ax

    def hvplot(self, **kwargs: Any) -> Any:
        """Generate an interactive curtain plot using hvPlot (Track B).

        Parameters
        ----------
        **kwargs : Any
            Keyword arguments passed to `hvplot()`.
            `rasterize=True` is recommended for high performance on large datasets.

        Returns
        -------
        holoviews.core.Element
            The interactive hvPlot object.
        """
        try:
            import hvplot.xarray  # noqa: F401
        except ImportError:
            raise ImportError(
                "hvplot is required for interactive plotting. Install it with 'pip install hvplot'."
            )

        plot_kwargs = {"x": self.x, "y": self.y}
        plot_kwargs.update(kwargs)

        return self.data.hvplot(**plot_kwargs)
