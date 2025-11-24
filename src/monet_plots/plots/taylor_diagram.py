
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import corrcoef
from .base import BasePlot
from .. import taylordiagram as td
from ..plot_utils import to_dataframe
from typing import Any, Union, List

class TaylorDiagramPlot(BasePlot):
    """Create a DataFrame-based Taylor diagram.

    A convenience wrapper for easily creating Taylor diagrams from DataFrames.
    """

    def __init__(self, df: Any, col1: str = "obs", col2: Union[str, List[str]] = "model", label1: str = "OBS", scale: float = 1.5, dia=None, *args, **kwargs):
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
            # Use self.fig which is created in BasePlot.__init__
            self.dia = td.TaylorDiagram(obsstd, scale=self.scale, fig=self.fig, rect=111, label=self.label1)
            # Add contours and grid for the new diagram
            contours = self.dia.add_contours(colors="0.5")
            plt.clabel(contours, inline=1, fontsize=10)
            plt.grid(alpha=0.5)

        # Loop through each model column and add it to the diagram
        for model_col in self.col2:
            model_std = self.df[model_col].std()
            cc = corrcoef(self.df[self.col1].values, self.df[model_col].values)[0, 1]
            self.dia.add_sample(model_std, cc, label=model_col, **kwargs)

        self.fig.legend(
            self.dia.samplePoints,
            [p.get_label() for p in self.dia.samplePoints],
            numpoints=1,
            loc='upper right'
        )
        self.fig.tight_layout()
        return self.dia
