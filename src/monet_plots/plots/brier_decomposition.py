import numpy as np
import pandas as pd
from typing import Optional, Any
from ..verification_metrics import compute_brier_score_components
from .base import BasePlot
from ..plot_utils import validate_dataframe, to_dataframe


class BrierScoreDecompositionPlot(BasePlot):
    """
    Brier Score Decomposition Plot.

    Visualizes the components of the Brier Score: Reliability,
    Resolution, and Uncertainty.
    BS = Reliability - Resolution + Uncertainty
    """

    def __init__(
        self,
        data: Any,
        reliability_col: str = "reliability",
        resolution_col: str = "resolution",
        uncertainty_col: str = "uncertainty",
        forecasts_col: Optional[str] = None,
        observations_col: Optional[str] = None,
        n_bins: int = 10,
        label_col: Optional[str] = None,
        fig=None,
        ax=None,
        **kwargs,
    ):
        super().__init__(fig=fig, ax=ax, **kwargs)
        self.data = to_dataframe(data)
        self.reliability_col = reliability_col
        self.resolution_col = resolution_col
        self.uncertainty_col = uncertainty_col
        self.forecasts_col = forecasts_col
        self.observations_col = observations_col
        self.n_bins = n_bins
        self.label_col = label_col

        if not (self.forecasts_col and self.observations_col):
            required_cols = [
                self.reliability_col,
                self.resolution_col,
                self.uncertainty_col,
            ]
            validate_dataframe(self.data, required_columns=required_cols)

    def _prepare_plot_data(self):
        """Helper to compute components if raw data provided or use existing ones."""
        if self.forecasts_col and self.observations_col:
            components_list = []
            if self.label_col:
                for name, group in self.data.groupby(self.label_col):
                    comps = compute_brier_score_components(
                        np.asarray(group[self.forecasts_col]),
                        np.asarray(group[self.observations_col]),
                        self.n_bins,
                    )
                    row = pd.Series(comps)
                    row["model"] = str(name)
                    components_list.append(row)
            else:
                comps = compute_brier_score_components(
                    np.asarray(self.data[self.forecasts_col]),
                    np.asarray(self.data[self.observations_col]),
                    self.n_bins,
                )
                row = pd.Series(comps)
                row["model"] = "Model"
                components_list.append(row)

            df_plot = pd.DataFrame(components_list)
            plot_label_col = "model"
        else:
            df_plot = self.data
            plot_label_col = self.label_col
        return df_plot, plot_label_col

    def plot(self, **kwargs):
        """
        Main plotting method.

        Args:
            **kwargs: Matplotlib kwargs.
        """
        title = kwargs.pop("title", "Brier Score Decomposition")
        # Compute components if raw data provided
        df_plot, plot_label_col = self._prepare_plot_data()

        # Prepare for plotting: make resolution negative for visualization
        df_plot = df_plot.copy()
        df_plot["resolution_plot"] = -df_plot[self.resolution_col]

        # Grouped bar plot
        if plot_label_col:
            labels = df_plot[plot_label_col].astype(str)
        else:
            labels = df_plot.index.astype(str)

        x = np.arange(len(labels))
        width = 0.25

        self.ax.bar(
            x - width,
            df_plot[self.reliability_col],
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
            df_plot[self.uncertainty_col],
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

    def hvplot(self, **kwargs):
        """Generate an interactive Brier Score decomposition plot using hvPlot."""
        import hvplot.pandas  # noqa: F401

        df_plot, plot_label_col = self._prepare_plot_data()

        df_plot_melted = df_plot.melt(
            id_vars=[plot_label_col] if plot_label_col else [],
            value_vars=[
                self.reliability_col,
                self.resolution_col,
                self.uncertainty_col,
            ],
            var_name="component",
            value_name="value",
        )

        plot_kwargs = {
            "x": plot_label_col if plot_label_col else "index",
            "y": "value",
            "by": "component",
            "kind": "bar",
            "title": "Brier Score Decomposition",
        }
        plot_kwargs.update(kwargs)

        return df_plot_melted.hvplot(**plot_kwargs)
