from textwrap import fill
from typing import Any, List, Union

import numpy as np
import pandas as pd

from .. import taylordiagram as td
from ..plot_utils import to_dataframe
from ..style import CB_MARKERS
from .base import BasePlot, figure_style


class TaylorDiagramPlot(BasePlot):
    """
    Create a DataFrame-based Taylor diagram (Unified API).

    Parameters
    ----------
    data : Any
        Data with var1 and var2 columns (DataFrame, xarray, etc.).
    var1 : str
        Name of the first variable (e.g., observations).
    var2 : str or list of str
        Name(s) of the second variable(s) (e.g., model predictions).
    label1 : str, optional
        Label for var1 (default: "OBS").
    scale : float, optional
        Scale factor for diagram.
    dia : TaylorDiagram, optional
        Existing diagram to add to.
    df : Any, optional
        Deprecated alias for ``data``.
    col1 : str, optional
        Legacy alias for var1.
    col2 : str or list, optional
        Legacy alias for var2.
    normalize : bool, optional
        Use standard deviation ratios and dimensionless centered RMS contours.
    show_error_table : bool, optional
        Show finite-pair bias, centered RMS difference, total RMSE, and counts.
        Raw geometry uses rows finite across all models to share one reference spread.
        Raw tables show both the row count used for errors and the available pair
        count. If common rows cannot define a point, errors use the available pairs.
    comparison_groups : sequence of mappings, optional
        Normalized groups with label, reference column, model columns, and optional unit.
    **kwargs : Any
        Additional keyword arguments.
    """

    def __init__(
        self,
        data: Any = None,
        var1: str = None,
        var2: Union[str, List[str]] = None,
        label1: str = "OBS",
        scale: float = 1.5,
        dia=None,
        *args,
        df: Any = None,
        col1: str = None,  # legacy alias
        col2: Union[str, List[str]] = None,  # legacy alias
        normalize=False,
        show_error_table=False,
        comparison_groups=None,
        **kwargs,
    ):
        if dia is not None:
            kwargs.setdefault("fig", dia.ax.figure)
            kwargs.setdefault("ax", dia._ax)
        super().__init__(*args, **kwargs)
        # Support legacy aliases
        self.var1 = var1 or col1 or "obs"
        if var2 is not None:
            self.var2 = [var2] if isinstance(var2, str) else var2
        elif col2 is not None:
            self.var2 = [col2] if isinstance(col2, str) else col2
        else:
            self.var2 = ["model"]

        if df is not None and data is None:
            data = df
        if comparison_groups is not None and not normalize:
            raise ValueError("Comparison groups require normalized Taylor coordinates.")
        self.normalize = normalize
        self.show_error_table = show_error_table
        self.groups = (
            comparison_groups
            if comparison_groups is not None
            else [{"label": label1, "reference": self.var1, "models": self.var2}]
        )
        self.data = to_dataframe(data)
        self.label1 = label1
        self.scale = scale
        self.dia = dia
        self.error_table = pd.DataFrame()
        self._table_axes = None
        self._diagnostic = None
        self._added_samples = []

    def _comparison(self, data, group, model):
        reference = group["reference"]
        pair = data[[reference, model]].to_numpy(dtype=float, na_value=np.nan)
        pair = pair[np.isfinite(pair).all(axis=1)]
        row = {
            "reference": group["label"],
            "model": model,
            "unit": group.get("unit", ""),
            "paired_count": len(pair),
            "bias": np.nan,
            "centered_rms": np.nan,
            "rmse": np.nan,
            "normalized_bias": np.nan,
            "normalized_centered_rms": np.nan,
            "normalized_rmse": np.nan,
            "status": "",
        }
        if not len(pair):
            row["status"] = "No finite pairs"
            return row, None
        obs, pred = pair.T
        diff = pred - obs
        bias = diff.mean()
        row.update(
            bias=bias,
            centered_rms=np.sqrt(np.mean((diff - bias) ** 2)),
            rmse=np.sqrt(np.mean(diff**2)),
        )
        ref_std = obs.std()
        if ref_std > 0:
            for name in ("bias", "centered_rms", "rmse"):
                row[f"normalized_{name}"] = row[name] / ref_std
        if len(pair) < 2 or not ref_std or not pred.std():
            row["status"] = "Undefined correlation"
            return row, None
        return row, (pred.std() / ref_std, np.corrcoef(obs, pred)[0, 1])

    @figure_style
    def plot(self, **kwargs):
        """Generate the Taylor diagram and optional paired error table."""
        rows, points = [], []
        use_paired_statistics = self.normalize or self.show_error_table
        legacy = None
        if not self.normalize:
            legacy = (
                self.data[[self.var1, *self.var2]]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
        if self.normalize:
            ref_std = 1.0
        else:
            ref_std = legacy[self.var1].std(ddof=0 if self.show_error_table else 1)
        for group in self.groups:
            for model in group["models"]:
                label = f"{group['label']}: {model}" if self.normalize else model
                if use_paired_statistics:
                    data = self.data if self.normalize else legacy
                    row, coords = self._comparison(data, group, model)
                    if not self.normalize:
                        paired, _ = self._comparison(self.data, group, model)
                        row["available_count"] = paired["paired_count"]
                        if coords is not None:
                            coords = (coords[0] * ref_std, coords[1])
                        elif paired["paired_count"] > row["paired_count"]:
                            row = {
                                **paired,
                                "available_count": paired["paired_count"],
                                "status": "No point on common finite rows",
                            }
                    rows.append(row)
                else:
                    obs = legacy[self.var1]
                    pred = legacy[model]
                    coords = (
                        (pred.std(), obs.corr(pred))
                        if len(obs) >= 2 and obs.std() > 0 and pred.std() > 0
                        else None
                    )
                points.append((label, coords))

        if self.dia is None:
            if np.isfinite(ref_std) and ref_std > 0:
                if self.ax is not None:
                    self.fig.delaxes(self.ax)
                self.dia = td.TaylorDiagram(
                    ref_std,
                    scale=self.scale,
                    fig=self.fig,
                    rect=111,
                    label="Reference" if self.normalize else self.label1,
                )
                self.ax = self.dia._ax
                self.ax.set_title(
                    "Standard deviation ratio"
                    if self.normalize
                    else "Standard deviation"
                )
                self.dia._add_managed_contours(
                    5,
                    {"colors": "0.5"},
                    {"inline": True, "fontsize": 8, "fmt": "cRMS %.2g"},
                )
            elif self.ax is not None:
                self.ax.set_axis_off()
        elif self.normalize and self.dia.refstd != 1:
            raise ValueError(
                "A normalized Taylor diagram requires unit reference spread."
            )

        if (
            self.dia is not None
            and self.show_error_table
            and not self.normalize
            and np.isfinite(ref_std)
            and ref_std > 0
            and not np.isclose(self.dia.refstd, ref_std)
        ):
            raise ValueError(
                "The shared diagram must use the same paired reference standard deviation."
            )

        if self.dia is not None:
            for sample in self._added_samples:
                sample.remove()
                self.dia.samplePoints.remove(sample)
            self._added_samples.clear()
            for label, coords in points:
                if coords is not None:
                    index = (len(self.dia.samplePoints) - 1) % len(CB_MARKERS)
                    options = {"marker": CB_MARKERS[index]}
                    if np.isclose(coords[0], self.dia.refstd) and np.isclose(
                        coords[1], 1
                    ):
                        options.update(markerfacecolor="none", markersize=18, zorder=11)
                    options.update(kwargs)
                    sample = self.dia.add_sample(*coords, label=label, **options)
                    self._added_samples.append(sample)
            self.fig.subplots_adjust(right=0.73)
            self.ax.legend(
                handles=self.dia.samplePoints,
                labels=[
                    fill(point.get_label(), width=22) for point in self.dia.samplePoints
                ],
                loc="upper left",
                bbox_to_anchor=(1.04, 1),
                numpoints=1,
            )

        unavailable = [label for label, coords in points if coords is None]
        if self._diagnostic is not None:
            self._diagnostic.remove()
            self._diagnostic = None
        if unavailable and not self.show_error_table:
            self._diagnostic = self.fig.text(
                0.02,
                0.02,
                "Undefined Taylor coordinates: " + ", ".join(unavailable),
                fontsize=8,
                wrap=True,
            )
        if rows:
            table = pd.DataFrame(rows)
            previous = getattr(self.dia, "_error_rows", self.error_table)
            self.error_table = pd.concat([previous, table], ignore_index=True)
            self.error_table = self.error_table.drop_duplicates(
                subset=["reference", "model", "unit"], keep="last", ignore_index=True
            )
            if self.dia is not None:
                self.dia._error_rows = self.error_table
        if self.show_error_table and rows:
            self._draw_error_table()
        return self.ax

    def _draw_error_table(self):
        axes = getattr(self.dia, "_error_table_axes", self._table_axes)
        if axes in self.fig.axes:
            axes.remove()
        columns = [
            "reference",
            "model",
            "paired_count",
            "bias",
            "centered_rms",
            "rmse",
            "unit",
        ]
        headers = ["Reference", "Model", "N", "Bias", "cRMS", "RMSE", "Unit"]
        if self.normalize:
            columns += ["normalized_bias", "normalized_centered_rms", "normalized_rmse"]
            headers += ["Bias / σ", "cRMS / σ", "RMSE / σ"]
        else:
            columns.insert(3, "available_count")
            headers[2:3] = ["Error N", "Pair N"]
        columns.append("status")
        headers.append("Status")
        cells = [
            [
                "—"
                if pd.isna(value)
                else f"{value:.4g}"
                if isinstance(value, float)
                else fill(str(value), width=15)
                if index in (0, 1)
                else str(value)
                for index, value in enumerate(row)
            ]
            for row in self.error_table[columns].itertuples(index=False, name=None)
        ]
        self.fig.subplots_adjust(bottom=0.37, right=0.73, top=0.90)
        self._table_axes = self.fig.add_axes([0.04, 0.025, 0.92, 0.23])
        self._table_axes.set_axis_off()
        if self.dia is not None:
            self.dia._error_table_axes = self._table_axes
        weights = [1.6, 1.6, 0.55, 0.75, 1.15, 0.75, 0.6]
        if self.normalize:
            weights += [1, 1, 1]
        else:
            weights[2:3] = [0.8, 0.75]
        weights.append(2 if self.error_table["status"].ne("").any() else 0.7)
        widths = np.asarray(weights) / sum(weights)
        table = self._table_axes.table(
            cellText=cells,
            colLabels=headers,
            cellLoc="center",
            bbox=[0, 0, 1, 1],
            colWidths=widths,
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        for (row_index, _), cell in table.get_celld().items():
            cell.set_edgecolor("0.82")
            cell.set_facecolor("#e9eff4" if row_index == 0 else "white")
