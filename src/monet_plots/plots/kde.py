"""Kernel density estimates drawn with Matplotlib."""

import warnings

import numpy as np
from dask.base import is_dask_collection
from matplotlib import colormaps
from matplotlib.colors import Normalize, is_color_like
from scipy.stats import gaussian_kde

from .base import BasePlot, figure_style


def _support(kde, values, cut, clip, size):
    widths = np.sqrt(np.diag(kde.covariance))
    return [
        np.linspace(
            max(row.min() - width * cut, low),
            min(row.max() + width * cut, high),
            size,
        )
        for row, width, (low, high) in zip(values, widths, clip, strict=True)
    ]


def _contour_levels(density, levels):
    """Translate probability proportions into increasing density thresholds."""
    descending = np.sort(density.ravel())[::-1]
    mass = np.cumsum(descending) / descending.sum()
    indices = np.searchsorted(mass, 1 - levels).clip(0, len(descending) - 1)
    return np.unique(descending[indices])


def _numeric_vector(values):
    if hasattr(values, "to_numpy"):
        try:
            return values.to_numpy(dtype=float, na_value=np.nan)
        except TypeError:
            pass
    return np.asarray(values, dtype=float)


class KDEPlot(BasePlot):
    """Draw a univariate KDE or bivariate probability contours.

    Use `var1` or `x` for a single named variable, or `x` and `y` for
    joint density. Inputs can also be array-like. The `df` argument is
    an alias for `data`.
    """

    def __init__(
        self,
        data=None,
        var1=None,
        title=None,
        label=None,
        *args,
        df=None,
        x=None,
        y=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.data = data if data is not None else df
        self.var1 = var1 if var1 is not None else x if x is not None else y
        self.var2 = y if var1 is None and x is not None else None
        self.title = title
        self.label = label

    @figure_style
    def plot(
        self,
        *,
        x=None,
        y=None,
        weights=None,
        hue=None,
        hue_order=None,
        palette=None,
        hue_norm=None,
        bw_method=None,
        bw_adjust=1,
        gridsize=200,
        cut=3,
        clip=None,
        cumulative=False,
        common_norm=True,
        common_grid=False,
        multiple="layer",
        fill=False,
        levels=10,
        thresh=0.05,
        cbar=False,
        cbar_ax=None,
        cbar_kws=None,
        legend=True,
        warn_singular=True,
        log_scale=None,
        **kwargs,
    ):
        """Fit Gaussian kernels and draw density curves or contours.

        Bandwidth uses SciPy's `gaussian_kde` convention. For joint densities,
        contour `levels` give probability proportions between zero and one.
        Palettes accept Matplotlib colormap names, colors, or group-to-color
        mappings. Extra drawing options use Matplotlib artist keywords.
        """
        x = self.var1 if x is None else x
        y = self.var2 if y is None else y
        data = self.data
        if is_dask_collection(data):
            available = getattr(data, "columns", getattr(data, "variables", ()))
            requested = (x, y, weights, hue)
            if all(v is None or isinstance(v, (str, int, tuple)) for v in requested):
                columns = list(dict.fromkeys(v for v in requested if v in available))
                if columns and (x is not None or y is not None):
                    data = data[columns]
            data = data.compute()

        def resolve(value):
            if isinstance(value, str):
                return data[value]
            if (
                isinstance(value, (int, tuple))
                and hasattr(data, "columns")
                and value in data.columns
            ):
                return data[value]
            return value

        vertical = x is None and y is not None
        joint = x is not None and y is not None
        variables = (x, y) if joint else (y if vertical else x,)
        weights = None if weights is None else _numeric_vector(resolve(weights))

        axis_labels = tuple(v if isinstance(v, str) else "" for v in variables)
        if x is None and y is None and data is None:
            groups = []
        elif x is None and y is None and hasattr(data, "columns"):
            groups = [
                (col, [_numeric_vector(data[col])], weights)
                for col in data.select_dtypes(include=np.number).columns
            ]
        elif hue is not None:
            categories = resolve(hue)
            observed = (
                categories.dropna() if hasattr(categories, "dropna") else categories
            )
            labels = list(dict.fromkeys(observed)) if hue_order is None else hue_order
            values = [_numeric_vector(resolve(v)) for v in variables]
            groups = []
            for label in labels:
                mask = np.equal(categories, label)
                if hasattr(mask, "fillna"):
                    mask = mask.fillna(False)
                mask = np.asarray(mask, dtype=bool)
                groups.append(
                    (
                        label,
                        [v[mask] for v in values],
                        None if weights is None else weights[mask],
                    )
                )
        else:
            values = [
                _numeric_vector(resolve(v) if v is not None else data)
                for v in variables
            ]
            if x is None and y is None and values[0].ndim == 2:
                groups = [(i, [v], weights) for i, v in enumerate(values[0].T)]
            else:
                groups = [(self.label, values, weights)]

        dimensions = 2 if joint else 1
        if multiple not in {"layer", "stack", "fill"}:
            raise ValueError("multiple must be 'layer', 'stack', or 'fill'")
        if gridsize < 2 or bw_adjust <= 0:
            raise ValueError(
                "gridsize must be at least two and bw_adjust must be positive"
            )

        bounds = (-np.inf, np.inf) if clip is None else clip
        if np.ndim(bounds) == 1:
            bounds = [bounds] * dimensions
        bounds = [
            (-np.inf if low is None else low, np.inf if high is None else high)
            for low, high in bounds
        ]
        if isinstance(log_scale, (tuple, list)):
            logs = log_scale if joint else (log_scale[int(vertical)],)
        elif log_scale is None:
            scales = (self.ax.get_xscale(), self.ax.get_yscale())
            logs = [
                scale == "log"
                for scale in (scales if joint else (scales[int(vertical)],))
            ]
        else:
            logs = [log_scale] * dimensions
        bases = [10 if base is True else base for base in logs]
        for i, base in enumerate(bases):
            if not base:
                continue
            (self.ax.set_yscale if vertical or i == 1 else self.ax.set_xscale)(
                "log", base=base
            )
            low, high = bounds[i]
            bounds[i] = (
                np.log(low) / np.log(base) if low > 0 else -np.inf,
                np.log(high) / np.log(base) if np.isfinite(high) else np.inf,
            )

        fitted = []
        for group, values, group_weights in groups:
            values = np.asarray(values)
            if values.ndim != 2:
                raise ValueError("KDE variables must be one-dimensional")
            with np.errstate(divide="ignore", invalid="ignore"):
                for i, base in enumerate(bases):
                    if base:
                        values[i] = np.log(values[i]) / np.log(base)
            keep = np.isfinite(values).all(axis=0)
            if group_weights is not None:
                keep &= np.isfinite(group_weights)
                group_weights = group_weights[keep]
            values = values[:, keep]
            if group_weights is not None and np.any(group_weights < 0):
                raise ValueError("weights must be nonnegative")
            if (
                group_weights is not None
                and group_weights.size
                and group_weights.sum() == 0
            ):
                raise ValueError("weights must have positive sum")
            try:
                kde = gaussian_kde(values, bw_method=bw_method, weights=group_weights)
            except (ValueError, np.linalg.LinAlgError):
                insufficient = values.shape[1] <= dimensions
                constant = values.size and np.any(np.ptp(values, axis=1) == 0)
                if not (insufficient or constant):
                    # A linear relation can make the joint covariance singular.
                    if (
                        np.linalg.matrix_rank(
                            values - values.mean(axis=1, keepdims=True)
                        )
                        == dimensions
                    ):
                        raise
                if warn_singular:
                    warnings.warn(
                        "KDE skipped for singular data.", UserWarning, stacklevel=2
                    )
                continue
            kde.set_bandwidth(kde.factor * bw_adjust)
            mass = values.shape[1] if group_weights is None else group_weights.sum()
            fitted.append((group, values, kde, mass))

        shared = None
        if fitted and not joint and (common_grid or multiple != "layer"):
            pooled = np.concatenate([values for _, values, _, _ in fitted], axis=1)
            kde = gaussian_kde(pooled, bw_method=bw_method)
            kde.set_bandwidth(kde.factor * bw_adjust)
            shared = _support(kde, pooled, cut, bounds, gridsize)

        curves = []
        total = sum(mass for _, _, _, mass in fitted)
        for group, values, kde, mass in fitted:
            support = (
                shared
                if shared is not None
                else _support(kde, values, cut, bounds, gridsize)
            )
            if cumulative:
                if joint:
                    grid = np.meshgrid(*support)
                    origin = [v[0] for v in support]
                    density = np.array(
                        [
                            kde.integrate_box(origin, point)
                            for point in zip(*(v.flat for v in grid), strict=True)
                        ]
                    ).reshape(grid[0].shape)
                else:
                    density = np.array(
                        [kde.integrate_box_1d(support[0][0], v) for v in support[0]]
                    )
            elif joint:
                grid = np.meshgrid(*support)
                density = kde([v.ravel() for v in grid]).reshape(grid[0].shape)
            else:
                density = kde(support[0])
            if common_norm and len(fitted) > 1:
                density *= mass / total
            coordinates = [
                base**v if base else v for base, v in zip(bases, support, strict=True)
            ]
            curves.append((group, coordinates, density))

        color = kwargs.pop("color", None)
        if hue_norm is not None and curves:
            norm = hue_norm if isinstance(hue_norm, Normalize) else Normalize(*hue_norm)
            cmap = colormaps[palette or "viridis"]
            colors = cmap(norm([group for group, _, _ in curves]))
        elif is_color_like(palette) and not (
            isinstance(palette, str) and palette in colormaps
        ):
            colors = [palette] * len(curves)
        elif isinstance(palette, str):
            colors = colormaps[palette](np.linspace(0.1, 0.9, len(curves)))
        elif isinstance(palette, dict):
            colors = [palette[group] for group, _, _ in curves]
        elif palette is None:
            colors = [
                color or (f"C{i % 10}" if len(curves) > 1 else None)
                for i in range(len(curves))
            ]
        else:
            colors = [palette[i % len(palette)] for i in range(len(curves))]

        if joint:
            proportions = (
                np.linspace(thresh, 1, levels)
                if isinstance(levels, int)
                else np.asarray(levels)
            )
            if np.any((proportions < 0) | (proportions > 1)):
                raise ValueError(
                    "levels must be probability proportions between zero and one"
                )
            density = (
                np.concatenate([z.ravel() for _, _, z in curves]) if curves else None
            )
            shared_levels = (
                _contour_levels(density, proportions)
                if common_norm and curves
                else None
            )
            for (group, (support_x, support_y), z), shade in zip(
                curves, colors, strict=True
            ):
                boundaries = (
                    shared_levels
                    if shared_levels is not None
                    else _contour_levels(z, proportions)
                )
                options = dict(kwargs)
                if (
                    shade is not None
                    and "cmap" not in options
                    and "colors" not in options
                ):
                    options["colors"] = [shade]
                draw = self.ax.contourf if fill else self.ax.contour
                if len(boundaries) >= (2 if fill else 1):
                    artist = draw(support_x, support_y, z, levels=boundaries, **options)
                    if cbar:
                        self.fig.colorbar(
                            artist, ax=self.ax, cax=cbar_ax, **(cbar_kws or {})
                        )
                    if group is not None and hue is not None:
                        self.ax.plot([], [], color=shade, label=str(group))
            self.ax.set(xlabel=axis_labels[0], ylabel=axis_labels[1])
        else:
            baseline = 0
            norm = (
                np.sum([z for _, _, z in curves], axis=0)
                if multiple == "fill"
                else None
            )
            for (group, (support,), density), shade in zip(curves, colors, strict=True):
                if norm is not None:
                    density = np.divide(
                        density, norm, out=np.zeros_like(density), where=norm != 0
                    )
                top = baseline + density
                group_label = str(group) if group is not None else None
                if fill or multiple != "layer":
                    draw = self.ax.fill_betweenx if vertical else self.ax.fill_between
                    draw(
                        support, baseline, top, label=group_label, color=shade, **kwargs
                    )
                else:
                    coordinates = (top, support) if vertical else (support, top)
                    self.ax.plot(*coordinates, label=group_label, color=shade, **kwargs)
                if multiple != "layer":
                    baseline = top
            labels = (
                ("Density", axis_labels[0]) if vertical else (axis_labels[0], "Density")
            )
            self.ax.set(xlabel=labels[0], ylabel=labels[1])
        if (
            legend
            and (hue is not None or len(curves) > 1)
            and self.ax.get_legend_handles_labels()[0]
        ):
            self.ax.legend(title=hue if isinstance(hue, str) else None)
        if self.title:
            self.ax.set_title(self.title)
        self.ax.spines[["top", "right"]].set_visible(False)
        return self.ax
