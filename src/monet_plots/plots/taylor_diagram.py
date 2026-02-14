import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import corrcoef
from .base import BasePlot
from .. import taylordiagram as td
from ..plot_utils import to_dataframe
from typing import Any, Union, List


class TaylorDiagramPlot(BasePlot):
    """Create a DataFrame-based Taylor diagram.

    A convenience wrapper for easily creating Taylor diagrams from DataFrames.
    """

    def __init__(
        self,
        df: Any,
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
            df (pd.DataFrame, np.ndarray, xr.Dataset, xr.DataArray): DataFrame with observation and model data.
            col1 (str): Column name for observations.
            col2 (str or list): Column name(s) for model predictions.
            label1 (str): Label for observations.
            scale (float): Scale factor for diagram.
            dia (TaylorDiagram, optional): Existing diagram to add to.
        """
        super().__init__(*args, **kwargs)
        self.col1 = col1
        if isinstance(col2, str):
            self.col2 = [col2]
        else:
            self.col2 = col2

        # Ensure all specified columns exist before proceeding
        required_cols = [self.col1] + self.col2
        self.df = to_dataframe(df).dropna(subset=required_cols)

        self.label1 = label1
        self.scale = scale
        self.dia = dia

    def plot(self, **kwargs):
        """Generate the Taylor diagram."""
        # If no diagram is provided, create a new one
        if self.dia is None:
            obsstd = self.df[self.col1].std()

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
        for model_col in self.col2:
            model_std = self.df[model_col].std()
            cc = corrcoef(self.df[self.col1].values, self.df[model_col].values)[0, 1]
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

        stats = []
        obs_std = self.df[self.col1].std()
        stats.append({"name": self.label1, "std": obs_std, "corr": 1.0})

        for model_col in self.col2:
            model_std = self.df[model_col].std()
            cc = np.corrcoef(self.df[self.col1].values, self.df[model_col].values)[0, 1]
            stats.append({"name": model_col, "std": model_std, "corr": cc})

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
