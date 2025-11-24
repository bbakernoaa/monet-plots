import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Union, List, Any
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

    def __init__(self, fig=None, ax=None, **kwargs):
        super().__init__(fig=fig, ax=ax, **kwargs)

    def plot(self,
             data: Any,
             x_col: str,
             y_col: str,
             val_col: str,
             sig_col: Optional[str] = None,
             cmap: str = 'RdYlGn',
             center: float = 0.0,
             **kwargs):
        """
        Main plotting method.

        Args:
            data (pd.DataFrame, np.ndarray, xr.Dataset, xr.DataArray): Long-format dataframe.
            x_col (str): Column for x-axis (Columns).
            y_col (str): Column for y-axis (Rows).
            val_col (str): Column for cell values (color).
            sig_col (str, optional): Column for significance (marker).
            cmap (str): Colormap.
            center (float): Center value for colormap divergence.
            **kwargs: Seaborn heatmap kwargs.
        """
        df = to_dataframe(data)
        validate_dataframe(df, required_columns=[x_col, y_col, val_col])

        # Pivot Data
        pivot_data = df.pivot(index=y_col, columns=x_col, values=val_col)

        # TDD Anchor: Test pivot structure

        # Plot Heatmap
        sns.heatmap(pivot_data, ax=self.ax, cmap=cmap, center=center, annot=True, fmt=".2f", cbar_kws={'label': 'Relative Performance'}, **kwargs)

        # Add Significance Markers
        if sig_col:
            pivot_sig = df.pivot(index=y_col, columns=x_col, values=sig_col)
            self._overlay_significance(pivot_data, pivot_sig)

        self.ax.set_xlabel(x_col.title())
        self.ax.set_ylabel(y_col.title())
        self.ax.tick_params(axis='x', rotation=45)
        self.ax.set_title('Performance Scorecard')

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
                    self.ax.text(j + 0.5, rows - i - 0.5, '*',
                               ha='center', va='center',
                               fontweight='bold', fontsize=12,
                               color='black', zorder=5)

# TDD Anchors:
# 1. test_pivot_logic: Verify long-to-wide conversion.
# 2. test_significance_overlay: Verify markers are placed only on significant cells.