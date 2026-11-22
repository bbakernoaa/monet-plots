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
        # Compute if raw data provided
        if forecasts_col and observations_col:
            df = df.loc[
                df[forecasts_col].between(0, 1) & df[observations_col].isin([0, 1])
            ]
            if climatology is None:
                climatology = float(df[observations_col].mean())
            bin_centers, obs_freq, bin_counts = compute_reliability_curve(
                np.asarray(df[forecasts_col]), np.asarray(df[observations_col]), n_bins
            )
            plot_data = pd.DataFrame(
                {x_col: bin_centers, y_col: obs_freq, "count": bin_counts}
            )
        else:
            validate_dataframe(df, required_columns=[x_col, y_col])
            plot_data = df

        if "count" in plot_data:
            plot_data = plot_data.assign(
                **{y_col: plot_data[y_col].where(plot_data["count"].ne(0))}
            )

        # Draw Reference Lines
        self.ax.plot([0, 1], [0, 1], "k--", label="Perfect Reliability")
        if climatology is not None and np.isfinite(climatology):
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
                self.ax.plot(group[x_col], group[y_col], marker="o", label=name, **k)
        else:
            k = kwargs.copy()
            label = k.pop("label", "Model")
            self.ax.plot(
                plot_data[x_col], plot_data[y_col], marker="o", label=label, **k
            )

        # Histogram Overlay (Sharpness)
        if show_hist and "count" in plot_data.columns:
            self._add_sharpness_histogram(plot_data, x_col)

        # Formatting
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_xlabel("Forecast Probability")
        self.ax.set_ylabel("Observed Relative Frequency")
        self.ax.set_aspect("equal")
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()

    def _draw_skill_regions(self, clim):
        """Shades areas where BSS > 0."""
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
