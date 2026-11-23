from math import prod
from typing import Any, Optional

import numpy as np
import pandas as pd
import xarray as xr
from dask import array as da
from dask import delayed, is_dask_collection

from ..plot_utils import to_dataframe, validate_dataframe
from ..verification_metrics import compute_reliability_curve
from .base import BasePlot, figure_style


def _summarize_bins(partition, edges):
    """Reduce one input partition to counts and sums for its occupied bins."""
    forecast = partition["forecast"]
    valid = forecast.between(0, 1) & partition["event"].isin([0, 1])
    valid &= partition["group"].notna()
    rows = partition.loc[valid].copy()
    rows["forecast"] = rows["forecast"].astype(float)
    rows["event"] = rows["event"].astype(np.int64)
    # Half-open bins include their lower edge, with 1 in the final bin.
    bins = np.searchsorted(edges, rows["forecast"].to_numpy(), side="right") - 1
    rows["bin"] = np.minimum(bins, len(edges) - 2)
    return rows.groupby(["group", "bin"], observed=True).agg(
        count=("forecast", "size"), forecast=("forecast", "sum"), event=("event", "sum")
    )


def _summarize_array_chunk(edges, forecast, event, group=0):
    rows = pd.DataFrame(
        {
            "forecast": np.ravel(forecast),
            "event": np.ravel(event),
            "group": np.ravel(group) if np.ndim(group) else group,
        }
    )
    return _summarize_bins(rows, edges)


def _summarize_dataframe_partition(partition, columns, edges):
    return _summarize_bins(
        pd.DataFrame(
            {
                "forecast": partition[columns[0]],
                "event": partition[columns[1]],
                "group": partition[columns[2]] if len(columns) == 3 else 0,
            }
        ),
        edges,
    )


def _merge_bin_summaries(parts):
    return pd.concat(parts).groupby(level=["group", "bin"], observed=True).sum()


def _lazy_bin_summary(data, columns, n_bins):
    edges = np.linspace(0, 1, n_bins + 1)
    if isinstance(data, xr.Dataset):
        available = data.variables
    else:
        available = data.columns
    missing = [column for column in columns if column not in available]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    repetitions = 1
    if isinstance(data, xr.Dataset):
        if any(size == 0 for size in data.sizes.values()):
            raise ValueError("DataFrame cannot be empty")
        # Match xarray's tabular broadcasting without flattening a whole field.
        arrays = xr.broadcast(*(data[column] for column in columns))
        # Dataset.to_dataframe repeats each row over unused dimensions.
        repetitions = prod(
            size for dim, size in data.sizes.items() if dim not in arrays[0].dims
        )
        arrays = da.broadcast_arrays(*(array.data for array in arrays))
        blocks = zip(*(array.to_delayed().ravel() for array in arrays), strict=True)
        partials = [delayed(_summarize_array_chunk)(edges, *block) for block in blocks]
    else:
        selected = data[list(dict.fromkeys(columns))]
        partials = [
            delayed(_summarize_dataframe_partition)(part, columns, edges)
            for part in selected.to_delayed()
        ]

    # Bound fan-in and keep raw partition outputs out of the final compute result.
    while len(partials) > 1:
        partials = [
            delayed(_merge_bin_summaries)(partials[i : i + 8])
            for i in range(0, len(partials), 8)
        ]
    return partials[0].compute() * repetitions


class ReliabilityDiagramPlot(BasePlot):
    """
    Reliability Diagram Plot (Attributes Diagram).

    Visualizes Observed Frequency vs Forecast Probability.

    Functional Requirements:
    1. Plot Observed Frequency (y-axis) vs Forecast Probability (x-axis).
    2. Draw "Perfect Reliability" diagonal (1:1).
    3. Show climatology and the no-skill boundary.
    4. Shade "Skill" areas (where Brier Skill Score > 0).
    5. Include inset histogram of forecast usage (Sharpness) if requested.

    Edge Cases:
    - Empty bins (no forecasts with that probability).
    - Climatology not provided (cannot draw skill regions correctly).
    """

    def __init__(self, fig=None, ax=None, **kwargs):
        super().__init__(fig=fig, ax=ax, **kwargs)

    @figure_style
    def plot(
        self,
        data: Any,
        x_col: str = "prob",
        y_col: str = "freq",
        forecasts_col: Optional[str] = None,
        observations_col: Optional[str] = None,
        n_bins: int = 10,
        climatology: Optional[float] = None,
        label_col: Optional[str] = None,
        show_hist: bool = False,
        **kwargs,
    ):
        """
        Main plotting method.

        Args:
            data: Input data. Raw forecasts in Dask-backed xarray datasets or Dask
                DataFrames are reduced to bin summaries before rendering.
            x_col (str): Forecast Probability bin center (for pre-binned).
            y_col (str): Observed Frequency in bin (for pre-binned).
            forecasts_col (str, optional): Column of raw forecast probabilities [0,1].
            observations_col (str, optional): Column of binary observations {0,1}.
            n_bins (int): Number of bins for reliability curve computation.
            climatology (Optional[float]): Sample climatology (mean(observations)).
            label_col (str, optional): Grouping column.
            show_hist (bool): Whether to show frequency of usage histogram.
            **kwargs: Matplotlib kwargs.
        """
        histogram_data = None
        if forecasts_col and observations_col:
            required = [forecasts_col, observations_col] + (
                [label_col] if label_col else []
            )
            if not isinstance(n_bins, (int, np.integer)) or n_bins <= 0:
                raise ValueError("n_bins must be a positive integer.")
            if (
                isinstance(data, xr.Dataset) or hasattr(data, "columns")
            ) and is_dask_collection(data):
                plot_data, histogram_data, sample_climatology = self._lazy_raw_data(
                    data, required, n_bins, label_col, x_col, y_col, show_hist
                )
            else:
                plot_data, histogram_data, sample_climatology = self._eager_raw_data(
                    data, required, n_bins, label_col, x_col, y_col, show_hist
                )
            if climatology is None:
                climatology = sample_climatology
        else:
            required = [x_col, y_col] + ([label_col] if label_col else [])
            plot_data = to_dataframe(data)
            validate_dataframe(plot_data, required_columns=required)

        if "count" in plot_data:
            plot_data = plot_data.assign(
                **{y_col: plot_data[y_col].where(plot_data["count"].ne(0))}
            )

        # Draw Reference Lines
        self.ax.plot([0, 1], [0, 1], "k--", label="Perfect Reliability")
        if climatology is not None:
            if not 0 <= climatology <= 1:
                raise ValueError(
                    "Climatology must be a probability between zero and one."
                )
            self.ax.axhline(
                climatology, color="gray", linestyle=":", label="Climatology"
            )
            self._draw_skill_regions(climatology)

        # Plot Data
        if label_col:
            for name, group in plot_data.groupby(label_col):
                # pop label from kwargs if it exists to avoid multiple values
                k = kwargs.copy()
                k.pop("label", None)
                self.ax.plot(
                    group[x_col], group[y_col], label=name, **{"marker": "o", **k}
                )
        else:
            k = kwargs.copy()
            label = k.pop("label", "Model")
            self.ax.plot(
                plot_data[x_col], plot_data[y_col], label=label, **{"marker": "o", **k}
            )

        # Histogram Overlay (Sharpness)
        if show_hist and "count" in plot_data.columns:
            if histogram_data is None:
                histogram_data = plot_data.groupby(x_col, as_index=False)["count"].sum()
            self._add_sharpness_histogram(histogram_data.dropna(subset=[x_col]), x_col)

        # Formatting
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_xlabel("Forecast Probability")
        self.ax.set_ylabel("Observed Relative Frequency")
        self.ax.set_aspect("equal")
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()

    @staticmethod
    def _lazy_raw_data(data, columns, n_bins, label_col, x_col, y_col, show_hist):
        summary = _lazy_bin_summary(data, columns, n_bins)
        if summary.empty:
            raise ValueError("No valid forecast probabilities and binary observations.")
        sample_climatology = float(summary["event"].sum() / summary["count"].sum())

        def format_bins(group):
            group = group.reindex(range(n_bins), fill_value=0)
            count = group["count"]
            denominator = count.where(count.ne(0))
            return pd.DataFrame(
                {
                    x_col: group["forecast"] / denominator,
                    y_col: group["event"] / denominator,
                    "count": count,
                }
            )

        pooled = summary.groupby(level="bin").sum()
        histogram = format_bins(pooled) if show_hist else None
        if label_col:
            plotted = pd.concat(
                [
                    format_bins(group.droplevel("group")).assign(**{label_col: label})
                    for label, group in summary.groupby(level="group", observed=True)
                ],
                ignore_index=True,
            )
        else:
            plotted = format_bins(pooled)
        return plotted, histogram, sample_climatology

    @staticmethod
    def _eager_raw_data(data, required, n_bins, label_col, x_col, y_col, show_hist):
        df = to_dataframe(data)
        validate_dataframe(df, required_columns=required)
        forecasts_col, observations_col = required[:2]
        df = df.loc[df[forecasts_col].between(0, 1) & df[observations_col].isin([0, 1])]
        if label_col:
            df = df.dropna(subset=[label_col])
        if df.empty:
            raise ValueError("No valid forecast probabilities and binary observations.")
        sample_climatology = float(df[observations_col].mean())

        def bin_forecasts(frame):
            x, frequency, count = compute_reliability_curve(
                frame[forecasts_col].to_numpy(dtype=float),
                frame[observations_col].to_numpy(dtype=float),
                n_bins,
            )
            return pd.DataFrame({x_col: x, y_col: frequency, "count": count})

        histogram_data = None
        if label_col:
            plot_data = pd.concat(
                [
                    bin_forecasts(group).assign(**{label_col: name})
                    for name, group in df.groupby(label_col, observed=True)
                ],
                ignore_index=True,
            )
            if show_hist:
                histogram_data = bin_forecasts(df)
        else:
            plot_data = bin_forecasts(df)
        return plot_data, histogram_data, sample_climatology

    def _draw_skill_regions(self, clim):
        """Show which bins improve on the reference event probability."""
        x = np.array([0, clim, 1])
        boundary = (x + clim) / 2
        self.ax.plot([0, 1], [clim / 2, (1 + clim) / 2], "k-.", label="No skill")
        self.ax.fill_between(
            x,
            0,
            boundary,
            where=x <= clim,
            interpolate=True,
            alpha=0.12,
            color="#009E73",
            label="Improvement over climatology",
        )
        self.ax.fill_between(
            x,
            boundary,
            1,
            where=x >= clim,
            interpolate=True,
            alpha=0.12,
            color="#009E73",
        )

    def _add_sharpness_histogram(self, data, x_col):
        """Adds a small inset axes for sharpness histogram."""
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        inset_ax = inset_axes(self.ax, width=1.5, height=1.2, loc="upper right")
        inset_ax.bar(data[x_col], data["count"], alpha=0.6, color="#0072B2", width=0.08)
        inset_ax.set_title("Sharpness")
        inset_ax.set_xlabel(x_col)
        inset_ax.set_ylabel("Count")
