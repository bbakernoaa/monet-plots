# src/monet_plots/plots/base.py
"""Base class for all plots, ensuring a consistent interface and style."""

from __future__ import annotations

import re
from functools import wraps
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
from matplotlib import cbook
from matplotlib.backends.backend_agg import RendererAgg

from ..style import FIGURE_STYLES, set_style

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.cm
    import matplotlib.colorbar


def figure_style(method):
    """Apply opt-in settings while drawing or exporting one figure."""

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if self._figure_style is None:
            return method(self, *args, **kwargs)
        with plt.rc_context(self._figure_style):
            return method(self, *args, **kwargs)

    return wrapper


def _wrap_title(title, width, renderer):
    """Wrap at word boundaries using the title's rendered font metrics."""
    font = title.get_fontproperties()
    wrapped = []
    for line in title.get_text().split("\n"):

        def measure(text):
            math = (
                "TeX"
                if title.get_usetex()
                else (title.get_parse_math() and cbook.is_math_text(text))
            )
            return renderer.get_text_width_height_descent(text, font, ismath=math)[0]

        if measure(line) <= width:
            wrapped.append(line)
            continue
        current = ""
        tokens = re.findall(r"(?:\$[^$]+\$|[^\s])+", line)
        for token in tokens:
            candidate = f"{current} {token}" if current else token
            if current and measure(candidate) > width:
                wrapped.append(current)
                current = token
            else:
                current = candidate
        wrapped.append(current)
    return "\n".join(wrapped)


class BasePlot:
    """Base class for all plots.

    Handles figure and axis creation, applies a consistent style,
    and provides a common interface for saving and closing plots.
    """

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
        self._figure_style = FIGURE_STYLES.get(style)
        if style and self._figure_style is None:
            set_style(style)

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
            with plt.rc_context(self._figure_style):
                self.fig, self.ax = plt.subplots(**kwargs)

    @figure_style
    def save(self, filename, **kwargs):
        """Save the plot and wrap panel titles to their axes widths.

        Args:
            filename (str): The name of the file to save the plot to.
            **kwargs: Additional keyword arguments for `savefig`.
        """
        titles = [
            (ax, title, title.get_text())
            for ax in self.fig.axes
            for title in (ax._left_title, ax.title, ax._right_title)
            if title.get_visible() and title.get_text()
        ]
        if not titles:
            self.fig.savefig(filename, **kwargs)
            return
        # Resolve current layout and fixed-aspect axes before measuring.
        self.fig.draw_without_rendering()
        renderer = RendererAgg(1, 1, self.fig.dpi)
        original_sizes = [title.get_fontsize() for _, title, _ in titles]
        original_top = self.fig.subplotpars.top
        positions = None
        try:
            for ax, title, original in titles:
                wrapped = _wrap_title(title, ax.bbox.width * 0.98, renderer)
                if wrapped != original:
                    title.set_text(wrapped)
            bounded_export = (
                kwargs.get("bbox_inches", plt.rcParams["savefig.bbox"]) is None
            )
            if bounded_export and self.fig.get_layout_engine() is None:
                height = self.fig.bbox.height
                padding = renderer.points_to_pixels(2)
                top_limit = self.fig.bbox.y1 - padding
                excess = max(
                    title.get_window_extent(renderer).y1 - top_limit
                    for _, title, _ in titles
                )
                if excess > 0:
                    positions = [
                        (
                            ax,
                            ax.get_position(original=True).frozen(),
                            ax.get_in_layout(),
                        )
                        for ax in self.fig.axes
                    ]
                    min_top = self.fig.subplotpars.bottom + 0.12
                    self.fig.subplots_adjust(
                        top=max(min_top, original_top - excess / height)
                    )
                    self.fig.draw_without_rendering()
                    for ax, title, original in titles:
                        title.set_text(original)
                        title.set_text(
                            _wrap_title(title, ax.bbox.width * 0.98, renderer)
                        )
                        while (
                            title.get_window_extent(renderer).y1 > top_limit
                            and title.get_fontsize() > 1
                        ):
                            title.set_fontsize(title.get_fontsize() * 0.85)
                            title.set_text(original)
                            title.set_text(
                                _wrap_title(title, ax.bbox.width * 0.98, renderer)
                            )
            self.fig.savefig(filename, **kwargs)
        finally:
            if positions is not None:
                self.fig.subplots_adjust(top=original_top)
                for ax, position, in_layout in positions:
                    ax.set_position(position)
                    ax.set_in_layout(in_layout)
            for (_, title, original), size in zip(titles, original_sizes, strict=True):
                if title.get_text() != original:
                    title.set_text(original)
                if title.get_fontsize() != size:
                    title.set_fontsize(size)

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
