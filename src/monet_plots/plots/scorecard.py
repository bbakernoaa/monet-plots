import pandas as pd
import seaborn as sns
from typing import Optional, Any
from .base import BasePlot
from ..plot_utils import validate_dataframe, to_dataframe


class ScorecardPlot(BasePlot):
    """
    Scorecard Plot.

    Heatmap table displaying performance metrics across multiple dimensions
    (e.g., Variables vs Lead Times), colored by performance relative to a baseline.

    Functional Requirements:
    1. Heatmap grid: Rows (Variables/Regions), Cols (Lead Times/Levels).
    2. Color cells based on statistic (e.g., Difference from Baseline, RMSE ratio).
    3. Annotate cells with symbols (+/-) or values indicating significance.
    4. Handle Green (Better) / Red (Worse) color schemes correctly.

    Edge Cases:
    - Missing data for some cells (show as white/gray).
    - Infinite values (clip or mask).
    """

    def __init__(
        self,
        data: Any,
        x_col: str,
        y_col: str,
        val_col: str,
        sig_col: Optional[str] = None,
        cmap: str = "RdYlGn",
        center: float = 0.0,
        fig=None,
        ax=None,
        **kwargs,
    ):
        super().__init__(fig=fig, ax=ax, **kwargs)
        self.data = to_dataframe(data)
        self.x_col = x_col
        self.y_col = y_col
        self.val_col = val_col
        self.sig_col = sig_col
        self.cmap = cmap
        self.center = center
        validate_dataframe(
            self.data, required_columns=[self.x_col, self.y_col, self.val_col]
        )

    def plot(self, **kwargs):
        """
        Main plotting method.

        Args:
            **kwargs: Seaborn heatmap kwargs.
        """
        # Pivot Data
        pivot_data = self.data.pivot(
            index=self.y_col, columns=self.x_col, values=self.val_col
        )

        # TDD Anchor: Test pivot structure

        # Plot Heatmap
        sns.heatmap(
            pivot_data,
            ax=self.ax,
            cmap=self.cmap,
            center=self.center,
            annot=True,
            fmt=".2f",
            cbar_kws={"label": "Relative Performance"},
            **kwargs,
        )

        # Add Significance Markers
        if self.sig_col:
            pivot_sig = self.data.pivot(
                index=self.y_col, columns=self.x_col, values=self.sig_col
            )
            self._overlay_significance(pivot_data, pivot_sig)

        self.ax.set_xlabel(self.x_col.title())
        self.ax.set_ylabel(self.y_col.title())
        self.ax.tick_params(axis="x", rotation=45)
        self.ax.set_title("Performance Scorecard")

    def _overlay_significance(self, data_grid, sig_grid):
        """
        Overlays markers for significant differences.

        Assumes sig_grid contains boolean or truthy values for significance.
        """
        rows, cols = data_grid.shape
        for i in range(rows):
            for j in range(cols):
                sig_val = sig_grid.iloc[i, j]
                if pd.notna(sig_val) and bool(sig_val):
                    # Position at center of cell
                    self.ax.text(
                        j + 0.5,
                        rows - i - 0.5,
                        "*",
                        ha="center",
                        va="center",
                        fontweight="bold",
                        fontsize=12,
                        color="black",
                        zorder=5,
                    )

    # TDD Anchors:
    # 1. test_pivot_logic: Verify long-to-wide conversion.
    # 2. test_significance_overlay: Verify markers are placed only on significant cells.

    def hvplot(self, **kwargs):
        """Generate an interactive scorecard plot using hvPlot."""
        import hvplot.pandas  # noqa: F401

        plot_kwargs = {
            "x": self.x_col,
            "y": self.y_col,
            "C": self.val_col,
            "kind": "heatmap",
            "cmap": self.cmap,
            "title": "Performance Scorecard",
            "hover_cols": [self.sig_col] if self.sig_col else [],
        }
        plot_kwargs.update(kwargs)

        return self.data.hvplot(**plot_kwargs)
