"""Categorical plots drawn with Matplotlib."""

from colorsys import hls_to_rgb, rgb_to_hls
from functools import partial
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import Colormap, LinearSegmentedColormap, Normalize, to_rgba
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

from .. import style
from ..plot_utils import to_dataframe


def _columns(data, names, value_name):
    if isinstance(data, xr.DataArray):
        data = data.to_dataset(name=value_name if data.name is None else data.name)
    if isinstance(data, xr.Dataset):
        arrays = xr.broadcast(*(data[name] for name in names))
        result = xr.Dataset(
            {name: array.variable for name, array in zip(names, arrays)}
        ).compute()
        return {name: result[name].to_numpy().ravel() for name in names}
    frame = to_dataframe(data)
    return {name: frame[name].array for name in names}


def _levels(values, order):
    if order is not None:
        return list(order)
    if isinstance(values, pd.Categorical):
        return list(values.categories)
    levels = pd.unique(values)
    levels = levels[pd.notna(levels)]
    if pd.api.types.is_numeric_dtype(levels):
        levels = np.sort(levels)
    return list(levels)


def _eq(values, level):
    equal = values == level
    if isinstance(equal, np.ndarray) and equal.dtype == bool:
        return equal
    return pd.array(equal, dtype="boolean").fillna(False).to_numpy(dtype=bool)


def _percentiles(values, level):
    edge = (100 - level) / 2
    return np.percentile(values, [edge, 100 - edge])


def _interval(values, center, estimator, errorbar, n_boot, seed, weights, units):
    if errorbar is None or len(values) < 2:
        return None
    if callable(errorbar):
        return errorbar(values)
    if isinstance(errorbar, str):
        errorbar = (errorbar, 95 if errorbar in {"ci", "pi"} else 1)
    method, level = errorbar
    if method == "pi":
        return _percentiles(values, level)
    if method in {"sd", "se"}:
        delta = values.std(ddof=1) * level
        if method == "se":
            delta /= np.sqrt(len(values))
        return center - delta, center + delta
    if method != "ci":
        raise ValueError("errorbar must be 'ci', 'pi', 'sd', 'se', a callable, or None")
    if n_boot < 1:
        raise ValueError("n_boot must be positive for confidence intervals")
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_boot, dtype=float)
    if units is not None:
        unit_rows = [
            np.flatnonzero(_eq(units, group)) for group in _levels(units, None)
        ]
        for i in range(n_boot):
            rows = np.concatenate(
                [
                    rng.choice(
                        unit_rows[group], size=len(unit_rows[group]), replace=True
                    )
                    for group in rng.integers(len(unit_rows), size=len(unit_rows))
                ]
            )
            estimates[i] = (
                estimator(values[rows])
                if weights is None
                else _weighted_mean(values[rows], weights[rows])
            )
        return _percentiles(estimates[np.isfinite(estimates)], level)
    batch = max(1, min(n_boot, 250_000 // len(values)))
    for start in range(0, n_boot, batch):
        count = min(batch, n_boot - start)
        rows = rng.integers(len(values), size=(count, len(values)))
        samples = values[rows]
        if weights is not None:
            result = _weighted_mean(samples, weights[rows])
        elif estimator is np.mean:
            result = samples.mean(axis=1)
        else:
            result = [estimator(sample) for sample in samples]
        estimates[start : start + count] = result
    return _percentiles(estimates[np.isfinite(estimates)], level)


def _weighted_mean(values, weights):
    numerator = (values * weights).sum(axis=-1)
    denominator = weights.sum(axis=-1)
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator != 0,
    )


def _colors(levels, palette, color, hue_norm=None):
    if hue_norm is not None and levels:
        values = np.asarray(levels, dtype=float)
        norm = hue_norm if isinstance(hue_norm, Normalize) else Normalize(*hue_norm)
        if isinstance(palette, dict):
            return [palette[level] for level in levels]
        if palette is None or isinstance(palette, (str, Colormap)):
            cmap = plt.get_cmap("viridis" if palette is None else palette)
        else:
            cmap = LinearSegmentedColormap.from_list("categorical_hue", palette)
        return cmap(norm(values))
    if isinstance(palette, dict):
        return [palette[level] for level in levels]
    if isinstance(palette, str):
        cmap = plt.get_cmap(palette)
        if hasattr(cmap, "colors") and len(cmap.colors) <= 20:
            palette = cmap.colors
        else:
            return cmap(np.linspace(0.15, 0.85, len(levels)))
    if palette is None:
        palette = (
            [color]
            if color is not None
            else plt.rcParams["axes.prop_cycle"].by_key()["color"]
        )
    return [palette[i % len(palette)] for i in range(len(levels))]


def _draw_bars(ax, groups, horizontal, options, inverse_value, inverse_category):
    estimator = options.pop("estimator", np.mean)
    if isinstance(estimator, str):
        estimator = getattr(np, estimator)
    errorbar = options.pop("errorbar", ("ci", 95))
    ci = options.pop("ci", "deprecated")
    if ci != "deprecated":
        errorbar = "sd" if ci == "sd" else None if ci is None else ("ci", ci)
    seed = options.pop("seed", None)
    n_boot = options.pop("n_boot", 1000)
    capsize = options.pop("capsize", 0)
    error_options = {"color": ".26", "linewidth": 1.5, **options.pop("err_kws", {})}
    for old, new in (("errcolor", "color"), ("errwidth", "linewidth")):
        if old in options:
            error_options[new] = options.pop(old)
    draw_bar = ax.barh if horizontal else ax.bar
    for position, width, color, values, _, weights, units in groups:
        if weights is not None:
            if estimator is not np.mean or (
                errorbar is not None
                and errorbar != "ci"
                and not (isinstance(errorbar, tuple) and errorbar[0] == "ci")
            ):
                raise ValueError("Weighted bars support the mean and CI only")
            center = _weighted_mean(values, weights)
            if not np.isfinite(center):
                continue
        else:
            center = estimator(values)
        if inverse_category is np.asarray:
            draw_bar(position, inverse_value(center), width, color=color, **options)
        else:
            lower, upper = inverse_category(
                [position - width / 2, position + width / 2]
            )
            draw_bar(
                lower,
                inverse_value(center),
                upper - lower,
                color=color,
                **{"align": "edge", **options},
            )
        limits = _interval(
            values, center, estimator, errorbar, n_boot, seed, weights, units
        )
        if limits is None:
            continue
        coordinates = [[position, position], limits]
        if capsize:
            cap = width * capsize / 2
            coordinates = [
                [
                    position - cap,
                    position + cap,
                    np.nan,
                    position,
                    position,
                    np.nan,
                    position - cap,
                    position + cap,
                ],
                [limits[0], limits[0], np.nan, *limits, np.nan, limits[1], limits[1]],
            ]
        coordinates = [inverse_category(coordinates[0]), inverse_value(coordinates[1])]
        ax.plot(*(coordinates[::-1] if horizontal else coordinates), **error_options)


def _draw_violins(ax, groups, horizontal, options, inverse_value, inverse_category):
    points = options.pop("gridsize", 100)
    bandwidth = options.pop("bw_method", options.pop("bw", "scott"))
    bw_adjust = options.pop("bw_adjust", 1)
    cut = options.pop("cut", 2)
    inner = options.pop("inner", "box")
    inner_options = {"color": ".26", **options.pop("inner_kws", {})}
    norm = options.pop("density_norm", options.pop("scale", "area"))
    common = options.pop("common_norm", False)
    if "scale_hue" in options:
        common = not options.pop("scale_hue")
    split = options.pop("split", False)
    fill = options.pop("fill", True)
    linecolor = options.pop("linecolor", ".26")
    edgecolor = options.pop("edgecolor", linecolor)
    if norm not in {"area", "count", "width"}:
        raise ValueError("density_norm must be 'area', 'count', or 'width'")
    if inner not in {None, "box", "quart", "quartile", "stick", "point"}:
        raise ValueError("inner must be 'box', 'quart', 'stick', 'point', or None")

    stats, peaks, counts = [], {}, {}
    for position, width, color, values, group, _, _ in groups:
        low, high = values.min(), values.max()
        support, density = np.array([low]), None
        key = None if common else group
        if low < high:
            kde = gaussian_kde(values, bw_method=bandwidth)
            kde.set_bandwidth(kde.factor * bw_adjust)
            pad = cut * float(np.sqrt(kde.covariance[0, 0]))
            support = np.linspace(low - pad, high + pad, points)
            density = kde(support)
            peaks[key] = max(peaks.get(key, 0), density.max())
        counts[key] = max(counts.get(key, 0), len(values))
        stats.append((position, width, color, values, group, key, support, density))

    draw = ax.fill_between if horizontal else ax.fill_betweenx

    def line(category, value, **kwargs):
        coords = (inverse_category(category), inverse_value(value))
        if horizontal:
            coords = coords[::-1]
        ax.plot(*coords, **kwargs)

    for position, width, color, values, group, key, support, density in stats:
        if density is None:
            line(
                [position - width / 2, position + width / 2],
                [support[0], support[0]],
                color=linecolor,
            )
            continue
        denominator = peaks[key] if norm == "area" else density.max()
        span = width * density / (2 * denominator)
        if norm == "count":
            span *= len(values) / counts[key]
        lower, upper = position - span, position + span
        if split:
            # Dodge centers are half a violin-width from the split center.
            center = position + (-1 if group % 2 else 1) * width / 2
            lower, upper = (
                (center - 2 * span, center)
                if group % 2 == 0
                else (center, center + 2 * span)
            )
            position = center
        draw(
            inverse_value(support),
            inverse_category(lower),
            inverse_category(upper),
            facecolor=color if fill else "none",
            edgecolor=edgecolor if fill else color,
            **options,
        )
        if inner in {"quart", "quartile", "stick"}:
            levels = np.percentile(values, [25, 50, 75]) if inner != "stick" else values
            for level in levels:
                line(
                    [
                        np.interp(level, support, lower),
                        np.interp(level, support, upper),
                    ],
                    [level, level],
                    **inner_options,
                )
        elif inner == "point":
            line(
                np.full(len(values), position),
                values,
                **{"linestyle": "none", "marker": ".", **inner_options},
            )
        elif inner == "box":
            q1, median, q3 = np.percentile(values, [25, 50, 75])
            cutoff = 1.5 * (q3 - q1)
            whiskers = values[(values >= q1 - cutoff) & (values <= q3 + cutoff)]
            line(
                [position, position], [whiskers.min(), whiskers.max()], **inner_options
            )
            line([position, position], [q1, q3], **{"linewidth": 3, **inner_options})
            line([position], [median], **{"marker": ".", **inner_options})


def categorical_plot(
    data,
    *,
    kind="bar",
    col_wrap=3,
    figsize=(15, 8),
    title=None,
    legend="auto",
    legend_labels=None,
    xlabel=None,
    ylabel=None,
    sharey=True,
    facet_kws=None,
    hue_norm=None,
    log_scale=None,
    **kwargs,
):
    """Draw categorical bars or violins with Matplotlib.

    Supply x and y as column, dimension, or coordinate names. By default x is
    categorical. Use orient="h" for a categorical y axis. Hue, row, and col
    specify grouping and faceting. Level order can be set using order,
    hue_order, row_order, and col_order. Bars show means and 95% bootstrap
    confidence intervals by default. Supply weights for a weighted mean or
    units for a clustered bootstrap. Violins use a Gaussian kernel density.

    Use hue_norm=(vmin, vmax) or a Matplotlib Normalize to map numeric hues
    to a colormap. log_scale accepts a boolean, a base, or an (x, y) pair.
    Statistics on the measurement axis are computed in log space when that
    axis is logarithmic. facet_kws controls shared axes, limits, subplot and
    GridSpec options, despining, margin titles, and legend placement.

    Returns the Matplotlib figure and NumPy array of axes.
    """
    if "x" not in kwargs or "y" not in kwargs:
        raise ValueError("x and y must be provided as keyword arguments")
    if kind not in {"bar", "violin"}:
        raise ValueError("kind must be 'bar' or 'violin'")
    x, y = kwargs.pop("x"), kwargs.pop("y")
    hue, row, col = (kwargs.pop(key, None) for key in ("hue", "row", "col"))
    orient = kwargs.pop("orient", None)
    weights_key, units_key = (kwargs.pop(key, None) for key in ("weights", "units"))
    if kind == "violin" and (weights_key is not None or units_key is not None):
        raise ValueError("Weights and sampling units apply only to bars")
    names = list(
        dict.fromkeys(
            key
            for key in (x, y, hue, row, col, weights_key, units_key)
            if key is not None
        )
    )
    columns = _columns(data, names, x if orient in {"h", "y"} else y)
    horizontal = orient in {"h", "y"} or (
        orient is None
        and pd.api.types.is_numeric_dtype(columns[x])
        and not pd.api.types.is_numeric_dtype(columns[y])
    )
    category, measurement = (y, x) if horizontal else (x, y)
    formatter = kwargs.pop("formatter", None)
    if formatter is not None:
        columns[category] = np.asarray(
            [
                formatter(value) if pd.notna(value) else None
                for value in columns[category]
            ]
        )
    category_levels = _levels(columns[category], kwargs.pop("order", None))
    native = kwargs.pop("native_scale", False)
    category_positions = np.arange(len(category_levels), dtype=float)
    if native and category_levels:
        if pd.api.types.is_datetime64_any_dtype(np.asarray(category_levels)):
            from matplotlib.dates import date2num

            category_positions = date2num(category_levels)
        elif pd.api.types.is_numeric_dtype(np.asarray(category_levels)):
            category_positions = np.asarray(category_levels, dtype=float)
        else:
            native = False
    log_x, log_y = (
        log_scale if isinstance(log_scale, (tuple, list)) else (log_scale, log_scale)
    )
    category_base, value_base = (log_y, log_x) if horizontal else (log_x, log_y)
    numeric_category = native and pd.api.types.is_numeric_dtype(
        np.asarray(category_levels)
    )
    if not isinstance(log_scale, (tuple, list)) and not numeric_category:
        category_base = None
    if any(
        base and base is not True and base <= 1 for base in (category_base, value_base)
    ):
        raise ValueError("Logarithm bases must be greater than one")
    category_base = 10 if category_base is True else category_base
    value_base = 10 if value_base is True else value_base
    inverse_category = np.asarray
    inverse_value = np.asarray
    if category_base:
        if not numeric_category:
            raise ValueError(
                "A logarithmic category axis requires native numeric positions"
            )
        positive = category_positions > 0
        category_levels = [
            level for level, keep in zip(category_levels, positive) if keep
        ]
        category_positions = np.log(category_positions[positive]) / np.log(
            category_base
        )
        inverse_category = partial(np.power, category_base)
    if value_base:
        inverse_value = partial(np.power, value_base)

    hue_levels, row_levels, col_levels = [
        _levels(columns[key], kwargs.pop(f"{role}_order", None))
        if key is not None
        else [None]
        for key, role in ((hue, "hue"), (row, "row"), (col, "col"))
    ]
    if not (row_levels and col_levels):
        raise ValueError("Facet variables must have at least one level")
    facets = list(product(row_levels, col_levels))
    nrows, ncols = len(row_levels), len(col_levels)
    wrap = row is None and col is not None and col_wrap is not None
    if wrap:
        if col_wrap < 1:
            raise ValueError("col_wrap must be positive")
        ncols = min(col_wrap, len(col_levels))
        nrows = (len(col_levels) + ncols - 1) // ncols
    width = kwargs.pop("width", 0.8)
    if native and len(category_positions) > 1:
        width *= np.diff(np.unique(category_positions)).min()
    gap = kwargs.pop("gap", 0)
    dodge = kwargs.pop("dodge", "auto")
    dodge = hue is not None if dodge == "auto" else dodge
    if dodge and hue_levels:
        width /= len(hue_levels)
    palette, color = kwargs.pop("palette", None), kwargs.pop("color", None)
    saturation = kwargs.pop("saturation", 0.75)

    def as_float(column):
        if isinstance(column, np.ndarray) and column.dtype.kind in "biuf":
            return column.astype(float, copy=False)
        numeric = pd.to_numeric(column, errors="coerce")
        return pd.array(numeric, dtype="Float64").to_numpy(dtype=float, na_value=np.nan)

    values = as_float(columns[measurement])
    weights = as_float(columns[weights_key]) if weights_key is not None else None
    units = columns[units_key] if units_key is not None else None
    finite = np.isfinite(values)
    if value_base:
        finite &= values > 0
        logged = np.full(values.shape, np.nan)
        np.log(values, out=logged, where=finite)
        values = logged / np.log(value_base)
    if weights is not None:
        finite &= np.isfinite(weights)
        if np.any(weights[finite] < 0):
            raise ValueError("Weights must be nonnegative")
    if units is not None:
        finite &= pd.notna(units)
    category_masks = [_eq(columns[category], level) for level in category_levels]
    hue_masks = (
        [_eq(columns[hue], level) for level in hue_levels]
        if hue is not None
        else [True]
    )
    draw = _draw_bars if kind == "bar" else _draw_violins

    facet_options = dict(facet_kws or {})
    xlim, ylim = (facet_options.pop(key, None) for key in ("xlim", "ylim"))
    despine = facet_options.pop("despine", False)
    margin_titles = facet_options.pop("margin_titles", False)
    legend_out = facet_options.pop("legend_out", True)
    facet_options.pop("dropna", None)  # Missing plotted values are already omitted.
    subplot_options = {
        "sharex": kwargs.pop("sharex", False),
        "sharey": sharey,
        **facet_options,
    }
    if hue is None and hue_norm is not None:
        raise ValueError("hue_norm requires a numeric hue variable")

    with plt.style.context(style.wiley_style):
        colors = _colors(hue_levels, palette, color, hue_norm)
        if saturation != 1:
            shades = []
            for shade in colors:
                r, g, b, alpha = to_rgba(shade)
                h, light, strength = rgb_to_hls(r, g, b)
                shades.append((*hls_to_rgb(h, light, strength * saturation), alpha))
            colors = shades
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=figsize,
            squeeze=False,
            **subplot_options,
        )
        for panel, (ax, (rlevel, clevel)) in enumerate(zip(axes.flat, facets)):
            if category_base:
                (ax.set_yscale if horizontal else ax.set_xscale)(
                    "log", base=category_base
                )
            if value_base:
                (ax.set_xscale if horizontal else ax.set_yscale)("log", base=value_base)
            if despine:
                ax.spines[["top", "right"]].set_visible(False)
            selected = finite.copy()
            for key, level in ((row, rlevel), (col, clevel)):
                if key is not None:
                    selected &= _eq(columns[key], level)
            groups = []
            for index, (hue_mask, shade) in enumerate(zip(hue_masks, colors)):
                offset = (index - (len(hue_levels) - 1) / 2) * width if dodge else 0
                for position, category_mask in zip(category_positions, category_masks):
                    mask = selected & hue_mask & category_mask
                    sample = values[mask]
                    if sample.size:
                        groups.append(
                            (
                                position + offset,
                                width * (1 - gap),
                                shade,
                                sample,
                                index,
                                None if weights is None else weights[mask],
                                None if units is None else units[mask],
                            )
                        )
            draw(ax, groups, horizontal, kwargs.copy(), inverse_value, inverse_category)
            tick_labels = [str(level) for level in category_levels]
            if native and pd.api.types.is_datetime64_any_dtype(
                np.asarray(category_levels)
            ):
                ax.yaxis_date() if horizontal else ax.xaxis_date()
            elif not native and horizontal and tick_labels:
                ax.set_yticks(range(len(tick_labels)), tick_labels)
                ax.set_ylim(len(tick_labels) - 0.5, -0.5)
            elif not native and tick_labels:
                ax.set_xticks(range(len(tick_labels)), tick_labels)
                ax.set_xlim(-0.5, len(tick_labels) - 0.5)
            ax.set_xlabel(x if xlabel is None else xlabel)
            ax.set_ylabel(y if ylabel is None else ylabel)
            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)
            if margin_titles and row is not None:
                if col is not None and panel < ncols:
                    ax.set_title(f"{col} = {clevel}", wrap=True)
                if panel % ncols == ncols - 1:
                    ax.text(
                        1.02,
                        0.5,
                        f"{row} = {rlevel}",
                        transform=ax.transAxes,
                        rotation=-90,
                        va="center",
                    )
            elif row is not None or col is not None:
                ax.set_title(
                    " | ".join(
                        f"{key} = {level}"
                        for key, level in ((row, rlevel), (col, clevel))
                        if key is not None
                    ),
                    wrap=True,
                )
        for ax in list(axes.flat)[len(facets) :]:
            ax.remove()
        if title is not None:
            fig.suptitle(title, wrap=True)
        if hue is not None and legend:
            labels = (
                [str(level) for level in hue_levels]
                if legend_labels is None
                else legend_labels
            )
            if len(labels) != len(hue_levels):
                raise ValueError("legend_labels must match the number of hue levels")
            legend_owner = fig if legend_out else axes.flat[0]
            legend_owner.legend(
                [Patch(facecolor=shade) for shade in colors], labels, title=hue
            )
    if wrap:
        axes = axes.ravel()[: len(facets)]
    return fig, axes


def categorical_timeseries(data, **kwargs):
    """Placeholder for categorical time series."""
    raise NotImplementedError("Categorical timeseries plots are not yet implemented.")
