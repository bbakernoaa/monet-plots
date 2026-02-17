import matplotlib.pyplot as plt
import pandas as pd
from .base import BasePlot
from .. import taylordiagram as td
from ..verification_metrics import compute_correlation
from typing import Any, Union, List


class TaylorDiagramPlot(BasePlot):
    """Create a DataFrame-based Taylor diagram.

    A convenience wrapper for easily creating Taylor diagrams from DataFrames.
    """

    def __init__(
        self,
        data: Any = None,
        col1: str = "obs",
        col2: Union[str, List[str]] = "model",
        label1: str = "OBS",
        scale: float = 1.5,
        dia=None,
        *args,
        **kwargs,
    ):
        """
        Initialize the plot with data and diagram settings.

        Args:
            data (pd.DataFrame, np.ndarray, xr.Dataset, xr.DataArray): Input data.
            col1 (str): Column name for observations.
            col2 (str or list): Column name(s) for model predictions.
            label1 (str): Label for observations.
            scale (float): Scale factor for diagram.
            dia (TaylorDiagram, optional): Existing diagram to add to.
        """
        if data is None and "df" in kwargs:
            data = kwargs.pop("df")
        super().__init__(*args, **kwargs)
        self.data = data
        self.col1 = col1
        if isinstance(col2, str):
            self.col2 = [col2]
        else:
            self.col2 = col2

        self.label1 = label1
        self.scale = scale
        self.dia = dia

    def plot(self, **kwargs):
        """Generate the Taylor diagram."""
        # Use centralized stats calculation
        stats_to_compute = {"obs_std": self.data[self.col1].std()}
        for col in self.col2:
            stats_to_compute[f"{col}_std"] = self.data[col].std()
            stats_to_compute[f"{col}_corr"] = compute_correlation(
                self.data[self.col1], self.data[col]
            )

        # Compute everything in one parallel operation if dask objects are present
        has_dask = any(
            hasattr(getattr(v, "data", None), "dask") for v in stats_to_compute.values()
        )
        if has_dask:
            import dask

            computed_stats = dask.compute(stats_to_compute)[0]
        else:
            # For non-dask objects, just extract values
            computed_stats = {
                k: v.item() if hasattr(v, "item") else v
                for k, v in stats_to_compute.items()
            }

        obsstd = computed_stats["obs_std"]

        # If no diagram is provided, create a new one
        if self.dia is None:
            # Remove the default axes created by BasePlot to avoid an extra empty plot
            if hasattr(self, "ax") and self.ax is not None:
                self.fig.delaxes(self.ax)

            # Use self.fig which is created in BasePlot.__init__
            self.dia = td.TaylorDiagram(
                obsstd, scale=self.scale, fig=self.fig, rect=111, label=self.label1
            )
            # Update self.ax to the one created by TaylorDiagram
            self.ax = self.dia._ax

            # Add contours for the new diagram
            contours = self.dia.add_contours(colors="0.5")
            plt.clabel(contours, inline=1, fontsize=10)

        # Loop through each model column and add it to the diagram
        # Use symbols by default for model data
        kwargs.setdefault("marker", "o")
        kwargs.setdefault("ls", "")

        for model_col in self.col2:
            model_std = computed_stats[f"{model_col}_std"]
            cc = computed_stats[f"{model_col}_corr"]
            self.dia.add_sample(model_std, cc, label=model_col, **kwargs)

        self.fig.legend(
            self.dia.samplePoints,
            [p.get_label() for p in self.dia.samplePoints],
            numpoints=1,
            loc="upper right",
        )
        self.fig.tight_layout()
        return self.dia

    def hvplot(self, **kwargs):
        """Generate a simplified interactive Taylor diagram using hvPlot."""
        import hvplot.pandas  # noqa: F401

        stats_to_compute = {"obs_std": self.data[self.col1].std()}
        for col in self.col2:
            stats_to_compute[f"{col}_std"] = self.data[col].std()
            stats_to_compute[f"{col}_corr"] = compute_correlation(
                self.data[self.col1], self.data[col]
            )

        has_dask = any(
            hasattr(getattr(v, "data", None), "dask") for v in stats_to_compute.values()
        )
        if has_dask:
            import dask

            computed_stats = dask.compute(stats_to_compute)[0]
        else:
            computed_stats = {
                k: v.item() if hasattr(v, "item") else v
                for k, v in stats_to_compute.items()
            }

        stats = []
        stats.append(
            {"name": self.label1, "std": computed_stats["obs_std"], "corr": 1.0}
        )

        for model_col in self.col2:
            stats.append(
                {
                    "name": model_col,
                    "std": computed_stats[f"{model_col}_std"],
                    "corr": computed_stats[f"{model_col}_corr"],
                }
            )

        df_stats = pd.DataFrame(stats)

        plot_kwargs = {
            "x": "std",
            "y": "corr",
            "kind": "scatter",
            "hover_cols": ["name"],
            "title": "Simplified Taylor Diagram (Std vs Corr)",
            "xlim": (0, df_stats["std"].max() * 1.1),
            "ylim": (-1.1, 1.1),
        }
        plot_kwargs.update(kwargs)

        return df_stats.hvplot(**plot_kwargs)
