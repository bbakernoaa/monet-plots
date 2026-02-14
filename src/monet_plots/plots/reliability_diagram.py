import numpy as np
import pandas as pd
from typing import Optional, Any
from ..verification_metrics import compute_reliability_curve
from .base import BasePlot
from ..plot_utils import validate_dataframe, to_dataframe


class ReliabilityDiagramPlot(BasePlot):
    """
    Reliability Diagram Plot (Attributes Diagram).

    Visualizes Observed Frequency vs Forecast Probability.

    Functional Requirements:
    1. Plot Observed Frequency (y-axis) vs Forecast Probability (x-axis).
    2. Draw "Perfect Reliability" diagonal (1:1).
    3. Draw "No Skill" line (horizontal at climatology/sample mean).
    4. Shade "Skill" areas (where Brier Skill Score > 0).
    5. Include inset histogram of forecast usage (Sharpness) if requested.

    Edge Cases:
    - Empty bins (no forecasts with that probability).
    - Climatology not provided (cannot draw skill regions correctly).
    """

    def __init__(
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
        fig=None,
        ax=None,
        **kwargs,
    ):
        super().__init__(fig=fig, ax=ax, **kwargs)
        self.data = to_dataframe(data)
        self.x_col = x_col
        self.y_col = y_col
        self.forecasts_col = forecasts_col
        self.observations_col = observations_col
        self.n_bins = n_bins
        self.climatology = climatology
        self.label_col = label_col
        self.show_hist = show_hist

        if not (self.forecasts_col and self.observations_col):
            validate_dataframe(self.data, required_columns=[self.x_col, self.y_col])

    def plot(self, **kwargs):
        """
        Main plotting method.

        Args:
            **kwargs: Matplotlib kwargs.
        """
        # Compute if raw data provided
        if self.forecasts_col and self.observations_col:
            if self.climatology is None:
                self.climatology = float(self.data[self.observations_col].mean())
            bin_centers, obs_freq, bin_counts = compute_reliability_curve(
                np.asarray(self.data[self.forecasts_col]),
                np.asarray(self.data[self.observations_col]),
                self.n_bins,
            )
            plot_data = pd.DataFrame(
                {self.x_col: bin_centers, self.y_col: obs_freq, "count": bin_counts}
            )
        else:
            plot_data = self.data

        # Draw Reference Lines
        self.ax.plot([0, 1], [0, 1], "k--", label="Perfect Reliability")
        if self.climatology is not None:
            self.ax.axhline(
                self.climatology, color="gray", linestyle=":", label="Climatology"
            )
            self._draw_skill_regions(self.climatology)

        # Plot Data
        if self.label_col:
            for name, group in plot_data.groupby(self.label_col):
                # pop label from kwargs if it exists to avoid multiple values
                k = kwargs.copy()
                k.pop("label", None)
                self.ax.plot(
                    group[self.x_col], group[self.y_col], marker="o", label=name, **k
                )
        else:
            k = kwargs.copy()
            label = k.pop("label", "Model")
            self.ax.plot(
                plot_data[self.x_col],
                plot_data[self.y_col],
                marker="o",
                label=label,
                **k,
            )

        # Histogram Overlay (Sharpness)
        if self.show_hist and "count" in plot_data.columns:
            self._add_sharpness_histogram(plot_data, self.x_col)

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
        x = np.linspace(0, 1, 100)
        y_no_skill = np.full_like(x, clim)
        y_perfect = x

        # Shade skill region (above no-skill towards perfect)
        self.ax.fill_between(
            x, y_no_skill, y_perfect, alpha=0.1, color="green", label="Skill Region"
        )

    def _add_sharpness_histogram(self, data, x_col):
        """Adds a small inset axes for sharpness histogram."""
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        inset_ax = inset_axes(self.ax, width=1.5, height=1.2, loc="upper right")
        inset_ax.bar(data[x_col], data["count"], alpha=0.5, color="blue", width=0.08)
        inset_ax.set_title("Sharpness")
        inset_ax.set_xlabel(x_col)
        inset_ax.set_ylabel("Count")

    def hvplot(self, **kwargs):
        """Generate an interactive reliability diagram using hvPlot."""
        import hvplot.pandas  # noqa: F401
        import holoviews as hv

        if self.forecasts_col and self.observations_col:
            if self.climatology is None:
                self.climatology = float(self.data[self.observations_col].mean())
            bin_centers, obs_freq, bin_counts = compute_reliability_curve(
                np.asarray(self.data[self.forecasts_col]),
                np.asarray(self.data[self.observations_col]),
                self.n_bins,
            )
            plot_data = pd.DataFrame(
                {
                    self.x_col: bin_centers,
                    self.y_col: obs_freq,
                    "count": bin_counts,
                }
            )
        else:
            plot_data = self.data

        plot_kwargs = {
            "x": self.x_col,
            "y": self.y_col,
            "kind": "scatter",
            "xlabel": "Forecast Probability",
            "ylabel": "Observed Relative Frequency",
            "title": "Reliability Diagram",
            "xlim": (0, 1),
            "ylim": (0, 1),
        }
        if self.label_col:
            plot_kwargs["by"] = self.label_col

        plot_kwargs.update(kwargs)

        p = plot_data.hvplot(**plot_kwargs)
        perfect = hv.Curve([(0, 0), (1, 1)]).opts(
            color="black", alpha=0.5, line_dash="dashed"
        )

        overlay = perfect * p

        if self.climatology is not None:
            clim_line = hv.HLine(self.climatology).opts(
                color="gray", alpha=0.5, line_dash="dotted"
            )
            overlay = overlay * clim_line

        return overlay
