from typing import Any, Optional

import numpy as np
import pandas as pd

from ..plot_utils import to_dataframe, validate_dataframe
from ..verification_metrics import compute_reliability_curve
from .base import BasePlot, figure_style


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
            data: Input data.
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
        df = to_dataframe(data)
        histogram_data = None
        if forecasts_col and observations_col:
            required = [forecasts_col, observations_col] + (
                [label_col] if label_col else []
            )
            validate_dataframe(df, required_columns=required)
            df = df.loc[
                df[forecasts_col].between(0, 1) & df[observations_col].isin([0, 1])
            ]
            if label_col:
                df = df.dropna(subset=[label_col])
            if df.empty:
                raise ValueError(
                    "No valid forecast probabilities and binary observations."
                )
            if climatology is None:
                climatology = float(df[observations_col].mean())

            def bin_forecasts(frame):
                x, frequency, count = compute_reliability_curve(
                    frame[forecasts_col].to_numpy(dtype=float),
                    frame[observations_col].to_numpy(dtype=float),
                    n_bins,
                )
                return pd.DataFrame({x_col: x, y_col: frequency, "count": count})

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
        else:
            required = [x_col, y_col] + ([label_col] if label_col else [])
            validate_dataframe(df, required_columns=required)
            plot_data = df

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
