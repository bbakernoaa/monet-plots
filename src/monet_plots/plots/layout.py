"""Compose heterogeneous MONET plots in one Matplotlib figure."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt


class PlotLayout:
    """Arrange different MONET plot classes in a shared figure.

    Parameters
    ----------
    shape : tuple[int, int]
        Number of subplot rows and columns.
    figsize : tuple[float, float], optional
        Matplotlib figure size.
    **kwargs
        Additional options passed to ``matplotlib.figure.Figure.add_gridspec``.

    Examples
    --------
    >>> layout = PlotLayout((1, 2))
    >>> sounding = layout.add(SkewTPlot, cell=(0, 0), plot_kwargs={"aspect": 80.5})
    >>> sounding.plot(pressure, temperature, dewpoint)
    >>> layout.add(SpatialPlot, cell=(0, 1), subplot_kwargs={"projection": ccrs.PlateCarree()})
    """

    def __init__(self, shape=(1, 1), *, figsize=None, **kwargs):
        if len(shape) != 2 or any(int(v) != v or v < 1 for v in shape):
            raise ValueError("shape must contain two positive integers")
        self.shape = tuple(map(int, shape))
        self.fig = plt.figure(figsize=figsize)
        self.grid = self.fig.add_gridspec(*self.shape, **kwargs)
        self.plots: dict[str, Any] = {}
        self.axes: dict[str, Any] = {}

    def add(
        self,
        plot_class,
        *args,
        cell,
        name=None,
        rowspan=1,
        colspan=1,
        subplot_kwargs=None,
        plot_kwargs=None,
        **constructor_kwargs,
    ):
        """Create a plot in a grid cell and return its plot instance.

        ``subplot_kwargs`` configures Matplotlib axes projections (including
        Cartopy and ``monet_skewt``); ``plot_kwargs`` are passed to the plot
        constructor. Call the returned plot's ``plot`` method to render it.
        """
        row, col = cell
        if (
            row < 0
            or col < 0
            or row + rowspan > self.shape[0]
            or col + colspan > self.shape[1]
        ):
            raise ValueError("cell and span must fit within the layout shape")
        key = name or f"{plot_class.__name__}_{len(self.plots) + 1}"
        if key in self.plots:
            raise ValueError(f"A plot named {key!r} already exists")
        ax = self.fig.add_subplot(
            self.grid[row : row + rowspan, col : col + colspan],
            **(subplot_kwargs or {}),
        )
        instance = plot_class(
            *args, fig=self.fig, ax=ax, **(plot_kwargs or {}), **constructor_kwargs
        )
        self.plots[key] = instance
        self.axes[key] = ax
        return instance

    def save(self, filename, **kwargs):
        """Save the composed figure."""
        self.fig.savefig(filename, **kwargs)

    def close(self):
        """Close the composed figure."""
        plt.close(self.fig)
