from typing import Any, Optional

import pandas as pd
import seaborn as sns

from ..plot_utils import to_dataframe, validate_dataframe
from .base import BasePlot


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

    def plot(
        self,
        data: Any,
        x_col: str,
        y_col: str,
        val_col: str,
        sig_col: Optional[str] = None,
        cmap: str = "RdYlGn",
        center: float = 0.0,
        annot_cols: Optional[list[str]] = None,
        cbar_labels: Optional[tuple[str, str]] = None,
        key_text: Optional[str] = None,
        **kwargs,
    ):
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
            annot_cols (list[str], optional): Columns to combine for cell annotations (e.g., ['mod', 'obs']).
            cbar_labels (tuple[str, str], optional): Labels for the left and right ends of the colorbar.
            key_text (str, optional): Text to display in a legend box at the top right.
            **kwargs: Seaborn heatmap kwargs.
        """
        df = to_dataframe(data).copy()
        validate_dataframe(df, required_columns=[x_col, y_col, val_col])

        # Extract title before passing kwargs to heatmap
        title = kwargs.pop("title", "Performance Scorecard")

        # Pivot Data
        pivot_data = df.pivot(index=y_col, columns=x_col, values=val_col)

        # Handle annotations
        if annot_cols:
            validate_dataframe(df, required_columns=annot_cols)
            # Create a combined annotation column
            df["_combined_annot"] = df[annot_cols[0]].map(
                lambda x: f"{x:.1f}" if pd.notna(x) else ""
            )
            for col in annot_cols[1:]:
                df["_combined_annot"] += " | " + df[col].map(
                    lambda x: f"{x:.1f}" if pd.notna(x) else ""
                )
            annot_data = df.pivot(index=y_col, columns=x_col, values="_combined_annot")
            kwargs["annot"] = annot_data
            kwargs["fmt"] = ""
        else:
            kwargs.setdefault("annot", True)
            kwargs.setdefault("fmt", ".2f")

        # Layout adjustments for WeatherMesh look
        cbar_ax = None
        is_weathermesh_layout = bool(cbar_labels or key_text)

        if is_weathermesh_layout:
            self.ax.set_title(title, pad=60)

            if cbar_labels:
                # Create a small axes for the colorbar at the top left
                cbar_ax = self.fig.add_axes([0.15, 0.85, 0.3, 0.02])
                kwargs["cbar_ax"] = cbar_ax
                kwargs["cbar_kws"] = kwargs.get("cbar_kws", {})
                kwargs["cbar_kws"]["orientation"] = "horizontal"

            if key_text:
                # Add a box at the top right
                self.fig.text(
                    0.85,
                    0.86,
                    key_text,
                    ha="right",
                    va="center",
                    bbox=dict(boxstyle="square", facecolor="white", edgecolor="black"),
                )
        else:
            self.ax.set_title(title)

        # Plot Heatmap
        kwargs.setdefault("linewidths", 0.5)
        kwargs.setdefault("linecolor", "lightgray")
        sns.heatmap(
            pivot_data,
            ax=self.ax,
            cmap=cmap,
            center=center,
            **kwargs,
        )

        # Post-process colorbar labels
        if cbar_ax and cbar_labels:
            cbar_ax.set_xticks([])
            cbar_ax.set_yticks([])
            self.fig.text(0.15, 0.83, cbar_labels[0], ha="left", va="top", fontsize=9)
            self.fig.text(0.45, 0.83, cbar_labels[1], ha="right", va="top", fontsize=9)

        # Add Significance Markers
        if sig_col:
            pivot_sig = df.pivot(index=y_col, columns=x_col, values=sig_col)
            self._overlay_significance(pivot_data, pivot_sig)

        self.ax.set_xlabel(x_col.title())
        if is_weathermesh_layout:
            self.ax.set_ylabel("")
            self.ax.tick_params(axis="x", rotation=0)
        else:
            self.ax.set_ylabel(y_col.title())
            self.ax.tick_params(axis="x", rotation=45)

        # Invert Y axis to have cities A-Z from top to bottom if desired,
        # but pivot might have already sorted them.

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
