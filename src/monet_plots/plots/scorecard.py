from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import pandas as pd
import seaborn as sns

from ..plot_utils import to_dataframe, validate_dataframe
from .base import BasePlot

if TYPE_CHECKING:
    import matplotlib.axes


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
        **kwargs: Any,
    ) -> matplotlib.axes.Axes:
        """
        Main plotting method.

        Parameters
        ----------
        data : Any
            Input data in long format. Can be a pandas DataFrame,
            xarray DataArray, xarray Dataset, or numpy ndarray.
        x_col : str
            Column name for x-axis (heatmap columns).
        y_col : str
            Column name for y-axis (heatmap rows).
        val_col : str
            Column name for cell values (color).
        sig_col : str, optional
            Column name for significance markers.
        cmap : str, optional
            Colormap name, by default "RdYlGn".
        center : float, optional
            Center value for colormap divergence, by default 0.0.
        **kwargs : Any
            Additional keyword arguments passed to seaborn.heatmap.

        Returns
        -------
        matplotlib.axes.Axes
            The axes object with the scorecard.
        """
        import xarray as xr

        # Track provenance if Xarray
        if isinstance(data, (xr.DataArray, xr.Dataset)):
            history = data.attrs.get("history", "")
            data.attrs["history"] = f"Generated ScorecardPlot; {history}"

        if isinstance(data, (xr.DataArray, xr.Dataset)):
            cols = [x_col, y_col, val_col]
            if sig_col:
                cols.append(sig_col)
            # Minimize data before conversion
            df = data[cols].to_dataframe().reset_index()
        else:
            df = to_dataframe(data)

        validate_dataframe(df, required_columns=[x_col, y_col, val_col])

        # Pivot Data
        pivot_data = df.pivot(index=y_col, columns=x_col, values=val_col)

        # TDD Anchor: Test pivot structure

        # Plot Heatmap
        sns.heatmap(
            pivot_data,
            ax=self.ax,
            cmap=cmap,
            center=center,
            annot=True,
            fmt=".2f",
            cbar_kws={"label": "Relative Performance"},
            **kwargs,
        )

        # Add Significance Markers
        if sig_col:
            pivot_sig = df.pivot(index=y_col, columns=x_col, values=sig_col)
            self._overlay_significance(pivot_data, pivot_sig)

        self.ax.set_xlabel(x_col.title())
        self.ax.set_ylabel(y_col.title())
        self.ax.tick_params(axis="x", rotation=45)
        self.ax.set_title("Performance Scorecard")

    def _overlay_significance(
        self, data_grid: pd.DataFrame, sig_grid: pd.DataFrame
    ) -> None:
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
