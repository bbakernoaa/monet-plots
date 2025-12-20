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

    def __init__(self, fig=None, ax=None, **kwargs):
        super().__init__(fig=fig, ax=ax, **kwargs)

    def plot(self,
             data: Any,
             x_col: str = 'prob',
             y_col: str = 'freq',
             forecasts_col: Optional[str] = None,
             observations_col: Optional[str] = None,
             n_bins: int = 10,
             climatology: Optional[float] = None,
             label_col: Optional[str] = None,
             show_hist: bool = False,
             **kwargs):
        """
        Main plotting method.

        Args:
            data (pd.DataFrame, np.ndarray, xr.Dataset, xr.DataArray): Either pre-binned reliability data or raw forecasts/observations.
            x_col (str): Forecast Probability bin center (for pre-binned).
            y_col (str): Observed Frequency in bin (for pre-binned).
            forecasts_col (str, optional): Column of raw forecast probabilities [0,1].
            observations_col (str, optional): Column of binary observations {0,1}.
            n_bins (int): Number of bins for reliability curve computation.
            climatology (Optional[float]): Sample climatology (mean(observations)) for skill lines.
            label_col (str, optional): Grouping column.
            show_hist (bool): Whether to show frequency of usage histogram.
            **kwargs: Matplotlib kwargs.
        """
        df = to_dataframe(data)
        # Compute if raw data provided
        if forecasts_col and observations_col:
            if climatology is None:
                climatology = float(df[observations_col].mean())
            bin_centers, obs_freq, bin_counts = compute_reliability_curve(
                np.asarray(df[forecasts_col]), np.asarray(
                    df[observations_col]), n_bins
            )
            plot_data = pd.DataFrame({
                x_col: bin_centers,
                y_col: obs_freq,
                'count': bin_counts
            })
        else:
            validate_dataframe(df, required_columns=[x_col, y_col])
            plot_data = df

        # Draw Reference Lines
        self.ax.plot([0, 1], [0, 1], 'k--', label='Perfect Reliability')
        if climatology is not None:
            self.ax.axhline(climatology, color='gray',
                            linestyle=':', label='Climatology')
            self._draw_skill_regions(climatology)

        # Plot Data
        if label_col:
            for name, group in plot_data.groupby(label_col):
                self.ax.plot(group[x_col], group[y_col],
                             marker='o', label=name, **kwargs)
            self.ax.legend(loc='best')
        else:
            self.ax.plot(plot_data[x_col], plot_data[y_col],
                         marker='o', label='Model', **kwargs)

        # Histogram Overlay (Sharpness)
        if show_hist and 'count' in plot_data.columns:
            self._add_sharpness_histogram(plot_data, x_col)

        # Formatting
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_xlabel("Forecast Probability")
        self.ax.set_ylabel("Observed Relative Frequency")
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()

    def _draw_skill_regions(self, clim):
        """
        Shades areas where BSS > 0.

        Shades the region between the perfect reliability line and the no-skill (climatology) line.
        """
        x = np.linspace(0, 1, 100)
        y_no_skill = np.full_like(x, clim)
        y_perfect = x

        # Shade skill region (above no-skill towards perfect)
        self.ax.fill_between(x, y_no_skill, y_perfect,
                             alpha=0.1, color='green', label='Skill Region')

        # TDD Anchor: Test geometry of skill regions.

    def _add_sharpness_histogram(self, data, x_col):
        """
        Adds a small inset axes for sharpness histogram.
        """
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        inset_ax = inset_axes(self.ax, width=1.5,
                              height=1.2, loc='upper right')
        inset_ax.bar(data[x_col], data['count'],
                     alpha=0.5, color='blue', width=0.08)
        inset_ax.set_title('Sharpness')
        inset_ax.set_xlabel(x_col)
        inset_ax.set_ylabel('Count')

# TDD Anchors:
# 1. test_skill_region_logic: Verify fill_between coordinates.
# 2. test_inset_histogram: Verify inset axes creation.
