from typing import Any, Optional

import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns

from ..plot_utils import to_dataframe, validate_dataframe
from ..style import FIGURE_STYLES
from .base import BasePlot, figure_style


class ScorecardPlot(BasePlot):
    """
    Scorecard Plot.

    Heatmap table displaying performance metrics across multiple dimensions
    (e.g., Variables vs Lead Times), colored by performance relative to a baseline.

    Functional Requirements:
    1. Heatmap grid: Rows (Variables/Regions), Cols (Lead Times/Levels).
    2. Color cells based on statistic (e.g., Difference from Baseline, RMSE ratio).
    3. Annotate cells with symbols (+/-) or values indicating significance.
    4. Label the direction of improvement when provided.

    Edge Cases:
    - Missing data for some cells (show as white/gray).
    - Infinite values (clip or mask).
    """

    _grouped_by_default = False

    def __init__(self, fig=None, ax=None, **kwargs):
        if fig is None and ax is None and kwargs.get("style") in FIGURE_STYLES:
            kwargs.setdefault("layout", "constrained")
        super().__init__(fig=fig, ax=ax, **kwargs)

    @figure_style
    def plot(
        self,
        data: Any,
        x_col: str,
        y_col: str,
        val_col: str,
        sig_col: Optional[str] = None,
        cmap: str = "RdBu_r",
        center: float = 0.0,
        annot_cols: Optional[list[str]] = None,
        cbar_labels: Optional[tuple[str, str]] = None,
        key_text: Optional[str] = None,
        *,
        row_group: str | None = None,
        col_group: str | None = None,
        baseline: str | None = None,
        units: str | None = None,
        lower_is_better: bool | None = None,
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
            row_group: Optional column grouping rows, shown with block dividers.
            col_group: Optional column grouping columns under a second header.
            baseline: Name of the reference for precomputed differences.
            units: Units of the cell values and color scale.
            lower_is_better: Set the interpretation of negative and positive changes.
            **kwargs: Seaborn heatmap kwargs.
        """
        df = to_dataframe(data).copy()
        validate_dataframe(df, required_columns=[x_col, y_col, val_col])

        # Extract title before passing kwargs to heatmap
        title = kwargs.pop("title", "Performance Scorecard")

        grouped_layout = (
            self._grouped_by_default
            or self._figure_style is not None
            or any(
                value is not None
                for value in (row_group, col_group, baseline, units, lower_is_better)
            )
        )
        if grouped_layout and not (cbar_labels or key_text):
            self._plot_grouped(
                df,
                x_col,
                y_col,
                val_col,
                sig_col,
                annot_cols,
                row_group,
                col_group,
                baseline,
                units,
                lower_is_better,
                title,
                cmap,
                center,
                kwargs,
            )
            return

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

    def _plot_grouped(
        self,
        df,
        x_col,
        y_col,
        val_col,
        sig_col,
        annot_cols,
        row_group,
        col_group,
        baseline,
        units,
        lower_is_better,
        title,
        cmap,
        center,
        options,
    ):
        row_keys = [row_group, y_col] if row_group else [y_col]
        col_keys = [col_group, x_col] if col_group else [x_col]
        validate_dataframe(
            df,
            required_columns=row_keys
            + col_keys
            + (annot_cols or [])
            + ([sig_col] if sig_col else []),
        )

        def ordered_index(keys):
            unique = df[keys].drop_duplicates()
            if len(keys) == 1:
                return pd.Index(unique[keys[0]])
            order = pd.Index(unique[keys[0]].unique())
            sorted_groups = unique.assign(_order=order.get_indexer(unique[keys[0]]))
            return pd.MultiIndex.from_frame(
                sorted_groups.sort_values("_order", kind="stable")[keys]
            )

        rows, columns = ordered_index(row_keys), ordered_index(col_keys)

        def grid(column):
            return df.pivot(index=row_keys, columns=col_keys, values=column).reindex(
                index=rows, columns=columns
            )

        values = grid(val_col).replace([np.inf, -np.inf], np.nan)
        raw = values.to_numpy(dtype=float, na_value=np.nan)
        finite = raw[np.isfinite(raw)]
        span = np.max(np.abs(finite - center)) if finite.size else 0
        low = options.pop("vmin", None)
        high = options.pop("vmax", None)
        span = (
            float(
                max(
                    span,
                    abs(low - center) if low is not None else 0,
                    abs(high - center) if high is not None else 0,
                )
            )
            or 1.0
        )
        low = center - span if low is None else low
        high = center + span if high is None else high
        norm = options.pop("norm", None)
        if norm is None:
            norm = mpl.colors.TwoSlopeNorm(vmin=low, vcenter=center, vmax=high)

        cell_format = options.pop("fmt", "+.1f")
        annotations = options.pop("annot", True)
        if annotations is True:
            if annot_cols:
                columns_to_show = [grid(column).to_numpy() for column in annot_cols]
                annotations = np.array(
                    [
                        " | ".join(
                            f"{column[i, j]:.1f}" if pd.notna(column[i, j]) else "—"
                            for column in columns_to_show
                        )
                        for i, j in np.ndindex(raw.shape)
                    ],
                    dtype=object,
                ).reshape(raw.shape)
            else:
                annotations = np.array(
                    [
                        format(value, cell_format).replace("-", "−")
                        for value in raw.flat
                    ],
                    dtype=object,
                ).reshape(raw.shape)
            if sig_col:
                flags = grid(sig_col).eq(True).fillna(False).to_numpy(dtype=bool)
                annotations[flags] = [text + "*" for text in annotations[flags]]
            cell_format = ""

        scale_label = f"Difference vs {baseline}" if baseline else val_col
        if units:
            scale_label += f" ({units})"
        if lower_is_better is not None:
            left, right = (
                ("better", "worse") if lower_is_better else ("worse", "better")
            )
            scale_label += f"  ·  Left: {left}  |  Right: {right}"
        under, over = np.any(finite < norm.vmin), np.any(finite > norm.vmax)
        extend = (
            "both"
            if under and over
            else "min"
            if under
            else "max"
            if over
            else "neither"
        )
        cbar = {
            "orientation": "horizontal",
            "pad": 0.09,
            "extend": extend,
            "label": scale_label,
            **options.pop("cbar_kws", {}),
        }
        options.setdefault("linewidths", 0.7)
        options.setdefault("linecolor", "white")
        palette = mpl.colormaps.get_cmap(cmap).with_extremes(bad="#e6e9ee")
        sns.heatmap(
            values,
            ax=self.ax,
            cmap=palette,
            norm=norm,
            annot=annotations,
            fmt=cell_format,
            cbar_kws=cbar,
            **options,
        )

        if annotations is not False:
            for i, j in zip(*np.where(~np.isfinite(raw)), strict=True):
                self.ax.text(
                    j + 0.5, i + 0.5, "—", ha="center", va="center", color="#526171"
                )
        self.ax.set_xticks(
            np.arange(len(columns)) + 0.5, columns.get_level_values(-1), rotation=0
        )
        row_labels = rows.get_level_values(-1).astype(str).tolist()
        if row_group:
            group_values = rows.get_level_values(0)
            starts = np.r_[0, np.flatnonzero(group_values[1:] != group_values[:-1]) + 1]
            for start in starts:
                row_labels[start] = f"{group_values[start]}  /  {row_labels[start]}"
                if start:
                    self.ax.axhline(start, color="#455467", linewidth=1.3)
        self.ax.set_yticks(np.arange(len(rows)) + 0.5, row_labels, rotation=0)
        self.ax.tick_params(axis="both", length=0)
        self.ax.xaxis.tick_top()
        if col_group:
            group_values = columns.get_level_values(0)
            bounds = np.r_[
                0,
                np.flatnonzero(group_values[1:] != group_values[:-1]) + 1,
                len(columns),
            ]
            header = self.ax.secondary_xaxis("top")
            header.set_xticks((bounds[:-1] + bounds[1:]) / 2, group_values[bounds[:-1]])
            header.tick_params(length=0, pad=15)
            header.spines["top"].set_visible(False)
            for label in header.get_xticklabels():
                label.set_fontweight("bold")
            for boundary in bounds[1:-1]:
                self.ax.axvline(boundary, color="#455467", linewidth=1.3)
        self.ax.set(xlabel="", ylabel="")
        heading = f"{title}\nRelative to {baseline}" if baseline else title
        if sig_col:
            heading += "  ·  * Significant"
        self.ax.set_title(
            heading, loc="left", pad=39 if col_group else 24, fontweight="bold"
        )

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
                        i + 0.5,
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
