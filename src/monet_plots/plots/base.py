# src/monet_plots/plots/base.py
"""Base class for all plots, ensuring a consistent interface and style."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

from ..style import _scoped_plot_method

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.cm
    import matplotlib.colorbar


class BasePlot:
    """Base class for all plots.

    Handles figure and axis creation, applies a consistent style,
    and provides a common interface for saving and closing plots.
    """

    def __init_subclass__(cls, **kwargs):
        """Keep styles local throughout construction, rendering and export."""
        super().__init_subclass__(**kwargs)
        for name, method in list(cls.__dict__.items()):
            if callable(method) and (
                name == "__init__"
                or name.startswith("plot")
                or name
                in {
                    "save",
                    "add_features",
                    "map_monet",
                    "map_dataframe",
                    "add_map_features",
                }
            ):
                setattr(cls, name, _scoped_plot_method(method))

    @_scoped_plot_method
    def __init__(self, fig=None, ax=None, style: str | None = "wiley", **kwargs):
        """Initializes the plot with a consistent style.

        If `fig` and `ax` are not provided, a new figure and axes
        are created.

        Args:
            fig (matplotlib.figure.Figure, optional): Figure to plot on.
            ax (matplotlib.axes.Axes, optional): Axes to plot on.
            style (str, optional): Style name to apply (e.g., 'wiley', 'paper').
                If None, no style is applied. Defaults to 'wiley'.
            **kwargs: Additional keyword arguments for `plt.subplots`.
        """
        self._style = style
        self.mappable = None
        self.colorbars = []
        if ax is not None and fig is not None and ax.figure is not fig:
            raise ValueError("fig must be the figure that owns ax.")

        if ax is not None:
            self.ax = ax
            if fig is not None:
                self.fig = fig
            else:
                self.fig = ax.figure
        elif fig is not None:
            self.fig = fig
            self.ax = None
        else:
            self.fig, self.ax = plt.subplots(**kwargs)

        self.axes = np.asarray([[self.ax]], dtype=object)

    @property
    def artists(self):
        """Rendered scalar artists and any observation overlays."""
        primary = getattr(self, "mappables", None)
        if primary is None:
            primary = [] if self.mappable is None else [self.mappable]
        return [*primary, *getattr(self, "overlays", [])]

    @_scoped_plot_method
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

    def add_logo(
        self,
        logo: str | Any | None = None,
        *,
        ax: matplotlib.axes.Axes | None = None,
        loc: str = "upper right",
        scale: float = 0.1,
        pad: float = 0.05,
        **kwargs: Any,
    ) -> Any:
        """Adds a logo to the plot.

        Parameters
        ----------
        logo : str or array-like, optional
            Path to the logo image, a URL, or a numpy array.
            If None, the default MONET logo is used.
        ax : matplotlib.axes.Axes, optional
            The axes to add the logo to. Defaults to `self.ax`.
        loc : str, optional
            Location of the logo ('upper right', 'upper left', 'lower right',
            'lower left', 'center'). Defaults to "upper right".
        scale : float, optional
            Scaling factor for the logo, by default 0.1.
        pad : float, optional
            Padding from the edge of the axes, by default 0.05.
        **kwargs : Any
            Additional keyword arguments passed to `AnnotationBbox`.

        Returns
        -------
        matplotlib.offsetbox.AnnotationBbox
            The added logo object.
        """
        import matplotlib.image as mpimg
        from matplotlib.offsetbox import AnnotationBbox, OffsetImage

        from ..plot_utils import get_logo_path

        if ax is None:
            ax = self.ax

        if logo is None:
            logo = get_logo_path()

        if isinstance(logo, str):
            if logo.startswith("http"):
                import io
                import urllib.request

                with urllib.request.urlopen(logo) as url:
                    f = io.BytesIO(url.read())
                img = mpimg.imread(f)
            else:
                img = mpimg.imread(logo)
        else:
            img = logo

        imagebox = OffsetImage(img, zoom=scale)
        imagebox.image.axes = ax

        # Mapping of location strings to axes fraction coordinates and box alignment
        loc_map = {
            "upper right": ((1 - pad, 1 - pad), (1, 1)),
            "upper left": ((pad, 1 - pad), (0, 1)),
            "lower right": ((1 - pad, pad), (1, 0)),
            "lower left": ((pad, pad), (0, 0)),
            "center": ((0.5, 0.5), (0.5, 0.5)),
        }

        if loc in loc_map:
            xy, box_alignment = loc_map[loc]
        else:
            # If loc is not a string in loc_map, assume it might be a coordinate
            # tuple, but for simplicity we default to upper right if it's invalid
            if isinstance(loc, tuple) and len(loc) == 2:
                xy = loc
                box_alignment = (0.5, 0.5)
            else:
                xy, box_alignment = loc_map["upper right"]

        ab = AnnotationBbox(
            imagebox,
            xy,
            xycoords="axes fraction",
            box_alignment=box_alignment,
            pad=0,
            frameon=False,
            **kwargs,
        )

        ax.add_artist(ab)
        return ab

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

        if loc not in {"right", "left", "top", "bottom"}:
            raise ValueError("loc must be right, left, top or bottom.")
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
        self.colorbars.append(cb)
        return cb
