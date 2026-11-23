"""Matplotlib heatmap rendering for the core diagnostic plots."""

from numbers import Integral

import matplotlib as mpl
import numpy as np


def _set_ticks(axis, labels, setting, inches):
    labels = np.asarray(labels)
    if setting is False:
        step = 0
    elif setting is True:
        step = 1
    elif isinstance(setting, Integral):
        step = int(setting)
    elif isinstance(setting, str) and setting == "auto":
        font_size = axis.get_major_ticks(1)[0].label1.get_fontsize()
        available = max(int(inches * 72 / font_size), 1)
        step = len(labels) // available + 1
    else:
        labels = np.asarray(setting)
        step = 1
    if step < 0:
        raise ValueError("Heatmap tick intervals must be non-negative")
    if step:
        axis.set_ticks(np.arange(0, len(labels), step) + 0.5, labels[::step])
    else:
        axis.set_ticks([])


def draw_heatmap(
    ax,
    values,
    *,
    xlabels=None,
    ylabels=None,
    cmap="viridis",
    center=None,
    vmin=None,
    vmax=None,
    norm=None,
    robust=False,
    mask=None,
    annot=False,
    fmt=".2g",
    annot_kws=None,
    xticklabels="auto",
    yticklabels="auto",
    square=False,
    cbar=True,
    cbar_ax=None,
    cbar_kws=None,
    linewidths=0,
    linecolor="white",
    **mesh_kws,
):
    """Draw a matrix with upper-left origin and options for labels and color scale."""
    values = np.ma.masked_invalid(np.ma.asarray(values, dtype=float))
    rows, columns = values.shape
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != values.shape:
            raise ValueError("Heatmap mask must have the same shape as the data")
        values = np.ma.masked_where(mask, values)
    annotation = values if annot is True else np.asarray(annot)
    if annot is not False and annot is not None and annotation.shape != values.shape:
        raise ValueError("Heatmap annotations must have the same shape as the data")

    finite = values.compressed()
    if finite.size:
        low, high = (
            np.percentile(finite, [2, 98]) if robust else (finite.min(), finite.max())
        )
    else:
        low, high = (0, 1) if center is None else (center - 1, center + 1)
    vmin = low if vmin is None else vmin
    vmax = high if vmax is None else vmax
    if isinstance(cmap, (list, tuple)):
        palette = mpl.colors.ListedColormap(cmap)
    else:
        palette = mpl.colormaps.get_cmap(cmap)

    if center is not None and norm is None:
        # Keep colorbar bounds at the observed range while putting the neutral
        # color at center, including when every observed value is on one side.
        span = max(abs(vmin - center), abs(vmax - center))
        if span:
            midpoint_scale = mpl.colors.Normalize(center - span, center + span)
            colors = palette(np.linspace(*midpoint_scale([vmin, vmax]), 256))
            palette = mpl.colors.ListedColormap(colors).with_extremes(
                bad=palette.get_bad(),
                under=palette.get_under(),
                over=palette.get_over(),
            )

    mesh_options = {
        "linewidths": linewidths,
        "edgecolors": linecolor,
        **mesh_kws,
    }
    if norm is None or isinstance(norm, str):
        mesh_options.update(vmin=vmin, vmax=vmax)
    mesh = ax.pcolormesh(values, cmap=palette, norm=norm, **mesh_options)
    ax.set(xlim=(0, columns), ylim=(rows, 0))
    if square:
        ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)

    if cbar:
        colorbar = ax.figure.colorbar(mesh, cax=cbar_ax, ax=ax, **(cbar_kws or {}))
        colorbar.outline.set_linewidth(0)
        if mesh.get_rasterized() and colorbar.solids is not None:
            colorbar.solids.set_rasterized(True)

    size = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
    _set_ticks(
        ax.xaxis,
        range(columns) if xlabels is None else xlabels,
        xticklabels,
        size.width,
    )
    _set_ticks(
        ax.yaxis, range(rows) if ylabels is None else ylabels, yticklabels, size.height
    )
    ax.tick_params(axis="y", labelrotation=0)

    if annot is not False and annot is not None:
        rgb = mesh.to_rgba(values)[..., :3]
        linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        luminance = linear @ np.array([0.2126, 0.7152, 0.0722])
        for row, column in zip(*np.where(~np.ma.getmaskarray(values)), strict=True):
            text_options = {
                "color": ".15" if luminance[row, column] > 0.408 else "white",
                "ha": "center",
                "va": "center",
                **(annot_kws or {}),
            }
            ax.text(
                column + 0.5,
                row + 0.5,
                format(annotation[row, column], fmt),
                **text_options,
            )
    return mesh
