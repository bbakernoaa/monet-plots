# src/monet_plots/plots/base.py
"""Base class for all plots, ensuring a consistent interface and style."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt

from ..style import wiley_style

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.cm
    import matplotlib.colorbar


class BasePlot:
    """Base class for all plots.

    Handles figure and axis creation, applies a consistent style,
    and provides a common interface for saving and closing plots.
    """

    def __init__(self, fig=None, ax=None, **kwargs):
        """Initializes the plot with a consistent style.

        If `fig` and `ax` are not provided, a new figure and axes
        are created.

        Args:
            fig (matplotlib.figure.Figure, optional): Figure to plot on.
            ax (matplotlib.axes.Axes, optional): Axes to plot on.
            **kwargs: Additional keyword arguments for `plt.subplots`.
        """
        plt.style.use(wiley_style)
        if ax is not None:
            self.ax = ax
            if fig is not None:
                self.fig = fig
            else:
                self.fig = ax.figure
        else:
            self.fig, self.ax = plt.subplots(**kwargs)

    def save(self, filename, **kwargs):
        """Saves the plot to a file.

        Args:
            filename (str): The name of the file to save the plot to.
            **kwargs: Additional keyword arguments for `savefig`.
        """
        self.fig.savefig(filename, **kwargs)

    def close(self):
        """Closes the plot figure."""
        plt.close(self.fig)

    def add_colorbar(
        self,
        mappable: matplotlib.cm.ScalarMappable,
        *,
        ax: matplotlib.axes.Axes | None = None,
        label: str | None = None,
        loc: str = "right",
        size: str = "5%",
        pad: float = 0.05,
        **kwargs: Any,
    ) -> matplotlib.colorbar.Colorbar:
        """Add a colorbar that matches the axes size.

        This method uses `inset_axes` to ensure the colorbar height (or width)
        matches the axes dimensions exactly, which is particularly useful for
        geospatial plots with fixed aspects.

        Parameters
        ----------
        mappable : matplotlib.cm.ScalarMappable
            The mappable object (e.g., from imshow, scatter, contourf).
        ax : matplotlib.axes.Axes, optional
            The axes to attach the colorbar to. Defaults to `self.ax`.
        label : str, optional
            Label for the colorbar, by default None.
        loc : str, optional
            Location of the colorbar ('right', 'left', 'top', 'bottom'),
            by default "right".
        size : str, optional
            Width (if vertical) or height (if horizontal) of the colorbar,
            as a percentage of the axes, by default "5%".
        pad : float, optional
            Padding between the axes and the colorbar, by default 0.05.
        **kwargs : Any
            Additional keyword arguments passed to `fig.colorbar`.

        Returns
        -------
        matplotlib.colorbar.Colorbar
            The created colorbar object.
        """
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        if ax is None:
            ax = self.ax

        orientation = "vertical" if loc in ["right", "left"] else "horizontal"

        # Determine anchor and position based on location
        if loc == "right":
            bbox_to_anchor = (1.0 + pad, 0.0, 1.0, 1.0)
            width, height = size, "100%"
        elif loc == "left":
            bbox_to_anchor = (-(float(size.strip("%")) / 100.0 + pad), 0.0, 1.0, 1.0)
            width, height = size, "100%"
        elif loc == "top":
            bbox_to_anchor = (0.0, 1.0 + pad, 1.0, 1.0)
            width, height = "100%", size
        else:  # bottom
            bbox_to_anchor = (0.0, -(float(size.strip("%")) / 100.0 + pad), 1.0, 1.0)
            width, height = "100%", size

        cax = inset_axes(
            ax,
            width=width,
            height=height,
            loc="lower left",
            bbox_to_anchor=bbox_to_anchor,
            bbox_transform=ax.transAxes,
            borderpad=0,
        )

        cb = self.fig.colorbar(mappable, cax=cax, orientation=orientation, **kwargs)

        if label:
            cb.set_label(label)

        return cb
