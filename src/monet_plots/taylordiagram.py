"""
Taylor diagram (Taylor, 2001) implementation.

A Taylor diagram is a graphical representation of how well a model simulates
an observed pattern. It provides a way to summarize multiple aspects of model
performance, including:
- Correlation coefficient
- Root-mean-square (RMS) difference
- The standard deviation ratio

Reference:
Taylor, K.E., 2001. Summarizing multiple aspects of model performance in a
single diagram. Journal of Geophysical Research, 106(D7), 7183-7192.
"""

import functools

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import MaxNLocator

__version__ = "Time-stamp: <2012-02-17 20:59:35 ycopin>"
__author__ = "Yannick Copin <yannick.copin@laposte.net>"

colors = ["#DA70D6", "#228B22", "#FA8072", "#FF1493"]


def _sns_context(f):
    """Decorator to apply seaborn color palette to a function."""

    @functools.wraps(f)
    def inner(*args, **kwargs):
        with sns.color_palette(colors):
            return f(*args, **kwargs)

    return inner


class TaylorDiagram:
    """
    :no-index:

    Taylor diagram for visualizing model performance metrics.

    The Taylor diagram displays multiple statistical metrics in a single plot:
    - The radial distance from the origin represents the standard deviation
    - The azimuthal position represents the correlation coefficient
    - The distance from the reference point represents the root-mean-square (RMS) difference

    This class creates a Taylor diagram in a polar plot, where:
    - r = standard deviation
    - θ = arccos(correlation coefficient)

    This provides a comprehensive view of how well a model pattern matches observations.
    """

    @_sns_context
    def __init__(self, refstd, scale=1.5, fig=None, rect=111, label="_"):
        """Initialize the Taylor diagram.

        Parameters
        ----------
        refstd : float
            The reference standard deviation (e.g., from observations or a reference model)
            that other models will be compared against.
        scale : float, default 1.5
            Initial radial limit as a multiple of refstd. The limit grows to fit samples.
            For example, if refstd=2 and scale=1.5, the maximum standard deviation
            displayed will be 3.0.
        fig : matplotlib.figure.Figure, optional
            Figure to use. If None, a new figure will be created.
        rect : int or tuple, default 111
            Subplot specification (nrows, ncols, index) or 3-digit integer where
            the digits represent nrows, ncols, and index in order.
        label : str, default "_"
            Label for the reference point. An underscore prefix makes the label not
            appear in the legend.
        """

        if not np.isfinite(refstd) or refstd <= 0:
            raise ValueError(
                "Reference standard deviation must be finite and positive."
            )
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Scale must be finite and positive.")
        self.refstd = refstd
        self.smin = 0
        self.smax = max(scale * refstd, refstd)
        self._theta_max = np.pi / 2
        self._contours = []
        fig = plt.figure() if fig is None else fig
        subplot = rect if isinstance(rect, tuple) else (rect,)
        self.ax = self._ax = fig.add_subplot(*subplot, projection="polar")
        self.ax.set_xlabel("Pearson correlation")
        self.ax.set_title("Standard deviation")
        self.ax.yaxis.set_major_locator(MaxNLocator(5))
        self.ax.grid(False)
        (reference,) = self.ax.plot(
            [0], [refstd], "r*", ms=14, label=label, zorder=10, clip_on=False
        )
        (self._reference_arc,) = self.ax.plot([], [], "k--", label="_")
        self.samplePoints = [reference]
        self._update_domain()

    def _update_domain(self):
        correlations = np.array([0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1])
        if self._theta_max > np.pi / 2:
            correlations = np.r_[-correlations[:0:-1], correlations]
        self.ax.set_thetamin(0)
        self.ax.set_thetamax(np.degrees(self._theta_max))
        self.ax.set_xticks(np.arccos(correlations), [f"{r:g}" for r in correlations])
        self.ax.set_ylim(self.smin, self.smax)
        theta = np.linspace(0, self._theta_max, 180)
        self._reference_arc.set_data(theta, np.full_like(theta, self.refstd))

    @property
    def samples(self):
        """Property to provide compatibility with tests expecting 'samples' attribute."""
        return self.samplePoints

    @_sns_context
    def add_sample(self, stddev, corrcoef, *args, **kwargs):
        """Add a sample point to the Taylor diagram.

        Parameters
        ----------
        stddev : float
            Standard deviation of the sample to add.
        corrcoef : float
            Correlation coefficient between the sample and reference (-1 to 1).
        *args
            Additional positional arguments passed to matplotlib's plot function.
        **kwargs
            Additional keyword arguments passed to matplotlib's plot function.
            Common options include 'marker', 'markersize', 'color', and 'label'.

        Returns
        -------
        matplotlib.lines.Line2D
            The line object representing the sample in the plot.

        Notes
        -----
        Points closer to the reference point indicate better agreement with
        the reference dataset.
        """
        if not np.isfinite(stddev) or stddev < 0:
            raise ValueError("Model standard deviation must be finite and nonnegative.")
        if not np.isfinite(corrcoef) or abs(corrcoef) > 1 + 8 * np.finfo(float).eps:
            raise ValueError("Pearson correlation must be finite and within [-1, 1].")
        theta = np.arccos(np.clip(corrcoef, -1, 1))
        if stddev >= self.smax or theta > self._theta_max:
            if stddev >= self.smax:
                self.smax = (
                    stddev * 1.05 if stddev <= np.finfo(float).max / 1.05 else stddev
                )
            if theta > self._theta_max:
                self._theta_max = np.pi
            self._update_domain()
            for index, (artist, levels, options, label_calls) in enumerate(
                self._contours
            ):
                if any(
                    args or kwargs.get("levels") is not None
                    for args, kwargs in label_calls
                ):
                    levels = artist.levels
                for label in artist.labelTexts:
                    if label in self.ax.texts:
                        label.remove()
                artist.labelTexts.clear()
                artist.remove()
                replacement = self._draw_contours(levels, options)
                for label_args, label_kwargs in label_calls:
                    replacement.clabel(*label_args, **label_kwargs)
                self._contours[index] = (replacement, levels, options, label_calls)

        # A single-point line is invisible without a marker; provide sensible defaults.
        kwargs.setdefault("marker", "o")
        kwargs.setdefault("ls", "")
        kwargs.setdefault("clip_on", False)
        (line,) = self.ax.plot(theta, stddev, *args, **kwargs)
        self.samplePoints.append(line)

        return line

    @_sns_context
    def add_contours(self, levels=5, **kwargs):
        """Add constant RMS difference contours to the Taylor diagram.

        Parameters
        ----------
        levels : int or array-like, default 5
            If an integer, it defines the number of equally-spaced contour levels.
            If array-like, it explicitly defines the contour levels.
        **kwargs
            Additional keyword arguments passed to matplotlib's contour function.
            Common options include 'colors', 'linewidths', and 'linestyles'.

        Returns
        -------
        matplotlib.contour.QuadContourSet
            The contour set created by the function.

        Notes
        -----
        These contours represent lines of constant RMS difference between the
        reference and sample datasets. They help visualize the combined effect
        of differences in standard deviation and correlation.
        """

        return self._add_managed_contours(levels, kwargs)

    def _add_managed_contours(self, levels, options, label_options=None):
        contours = self._draw_contours(levels, options)
        label_calls = []
        clabel = contours.clabel

        @functools.wraps(clabel)
        def remember_labels(*args, **kwargs):
            count = len(contours.labelTexts)
            labels = clabel(*args, **kwargs)
            if kwargs.get("manual") is not None and kwargs["manual"] is not False:
                kwargs["manual"] = [label.get_position() for label in labels[count:]]
            label_calls.append((args, kwargs))
            return labels

        # Axes.clabel and pyplot.clabel both dispatch to the contour set's method.
        contours.clabel = remember_labels
        if label_options is not None:
            self.ax.clabel(contours, **label_options)
        self._contours.append((contours, levels, options, label_calls))
        return contours

    def _draw_contours(self, levels, options):
        rs, ts = np.meshgrid(
            np.linspace(self.smin, self.smax, 100),
            np.linspace(0, self._theta_max, 180),
        )
        rms = np.hypot(rs * np.cos(ts) - self.refstd, rs * np.sin(ts))
        return self.ax.contour(ts, rs, rms, levels, **options)


if __name__ == "__main__":
    # Reference dataset
    x = np.linspace(0, 4 * np.pi, 100)
    data = np.sin(x)
    refstd = data.std(ddof=1)  # Reference standard deviation

    # Models
    m1 = data + 0.2 * np.random.randn(len(x))  # Model 1
    m2 = 0.8 * data + 0.1 * np.random.randn(len(x))  # Model 2
    m3 = np.sin(x - np.pi / 10)  # Model 3

    # Compute stddev and correlation coefficient of models
    samples = np.array(
        [[m.std(ddof=1), np.corrcoef(data, m)[0, 1]] for m in (m1, m2, m3)]
    )

    fig = plt.figure(figsize=(10, 4))

    # Taylor diagram
    dia = TaylorDiagram(refstd, fig=fig, rect=122, label="Reference")
    # colors_ = plt.matplotlib.cm.jet(np.linspace(0, 1, len(samples)))
    colors_ = [None] * len(samples)
    with sns.color_palette(colors):
        ax1 = fig.add_subplot(1, 2, 1, xlabel="X", ylabel="Y")
        ax1.plot(x, data, "ko", label="Data")
        for i, m in enumerate([m1, m2, m3]):
            ax1.plot(x, m, c=colors_[i], label="Model %d" % (i + 1))
    ax1.legend(numpoints=1, prop=dict(size="small"), loc="best")

    # Add samples to Taylor diagram
    for i, (stddev, corrcoef) in enumerate(samples):
        dia.add_sample(
            stddev,
            corrcoef,
            marker="s",
            ls="",
            c=colors_[i],
            label="Model %d" % (i + 1),
        )

    # Add RMS contours, and label them
    contours = dia.add_contours(colors="0.5")
    plt.clabel(contours, inline=1, fontsize=10)

    # Add a figure legend
    fig.legend(
        dia.samplePoints,
        [p.get_label() for p in dia.samplePoints],
        numpoints=1,
        prop=dict(size="small"),
        loc="upper right",
    )

    fig.tight_layout()

    plt.show()
