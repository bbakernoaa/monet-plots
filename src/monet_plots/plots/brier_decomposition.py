from typing import Any, Optional

import numpy as np
import pandas as pd

from ..plot_utils import to_dataframe, validate_dataframe
from ..verification_metrics import compute_brier_score_components
from .base import BasePlot


class BrierScoreDecompositionPlot(BasePlot):
    """
    Brier Score Decomposition Plot.

    Visualizes the components of the Brier Score: Reliability,
    Resolution, and Uncertainty.
    BS = Reliability - Resolution + Uncertainty
    """

    def __init__(self, fig=None, ax=None, **kwargs):
        super().__init__(fig=fig, ax=ax, **kwargs)

    def plot(
        self,
        data: Any,
        reliability_col: str = "reliability",
        resolution_col: str = "resolution",
        uncertainty_col: str = "uncertainty",
        forecasts_col: Optional[str] = None,
        observations_col: Optional[str] = None,
        n_bins: int = 10,
        label_col: Optional[str] = None,
        **kwargs,
    ):
        """
        Main plotting method.

        Args:
            data: Input data.
            reliability_col/resolution_col/uncertainty_col (str):
                Pre-computed component columns.
            forecasts_col/observations_col (str, optional):
                Raw forecast probabilities and binary observations.
            n_bins (int): Bins for decomposition if raw data.
            label_col (str, optional): Grouping column.
            **kwargs: Matplotlib kwargs.
        """
        title = kwargs.pop("title", "Brier Score Decomposition")
        df = to_dataframe(data)
        # Compute components if raw data provided
        if forecasts_col and observations_col:
            components_list = []
            if label_col:
                for name, group in df.groupby(label_col):
                    comps = compute_brier_score_components(
                        np.asarray(group[forecasts_col]),
                        np.asarray(group[observations_col]),
                        n_bins,
                    )
                    row = pd.Series(comps)
                    row["model"] = str(name)
                    components_list.append(row)
            else:
                comps = compute_brier_score_components(
                    np.asarray(df[forecasts_col]),
                    np.asarray(df[observations_col]),
                    n_bins,
                )
                row = pd.Series(comps)
                row["model"] = "Model"
                components_list.append(row)

            df_plot = pd.DataFrame(components_list)
            plot_label_col = "model"
        else:
            required_cols = [reliability_col, resolution_col, uncertainty_col]
            validate_dataframe(df, required_columns=required_cols)
            df_plot = df
            plot_label_col = label_col

        # Prepare for plotting: make resolution negative for visualization
        df_plot = df_plot.copy()
        df_plot["resolution_plot"] = -df_plot[resolution_col]

        # Grouped bar plot
        if plot_label_col:
            labels = df_plot[plot_label_col].astype(str)
        else:
            labels = df_plot.index.astype(str)

        x = np.arange(len(labels))
        width = 0.25

        self.ax.bar(
            x - width,
            df_plot[reliability_col],
            width,
            label="Reliability",
            color="red",
            alpha=0.8,
            **kwargs,
        )
        self.ax.bar(
            x,
            df_plot["resolution_plot"],
            width,
            label="Resolution (-)",
            color="green",
            alpha=0.8,
            **kwargs,
        )
        self.ax.bar(
            x + width,
            df_plot[uncertainty_col],
            width,
            label="Uncertainty",
            color="blue",
            alpha=0.8,
            **kwargs,
        )

        # Total Brier Score as line on top if available
        if "brier_score" in df_plot.columns:
            self.ax.plot(
                x,
                df_plot["brier_score"],
                "ko-",
                linewidth=2,
                markersize=6,
                label="Brier Score",
            )

        self.ax.set_xticks(x)
        self.ax.set_xticklabels(labels, rotation=45, ha="right")
        self.ax.legend(loc="best")
        self.ax.set_ylabel("Brier Score Components")
        self.ax.set_title(title)
        self.ax.grid(True, alpha=0.3)
