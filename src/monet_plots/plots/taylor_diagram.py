from typing import Any, List, Union

import matplotlib.pyplot as plt
from numpy import corrcoef

from .. import taylordiagram as td
from ..plot_utils import to_dataframe
from .base import BasePlot


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
        **kwargs,
    ):
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
        # Ensure all specified columns exist before proceeding
        required_cols = [self.var1] + self.var2
        self.data = to_dataframe(data).dropna(subset=required_cols)

        self.label1 = label1
        self.scale = scale
        self.dia = dia

    def plot(self, **kwargs):
        """Generate the Taylor diagram."""
        # If no diagram is provided, create a new one
        if self.dia is None:
            obsstd = self.data[self.col1].std()

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
        for model_col in self.var2:
            model_std = self.data[model_col].std()
            cc = corrcoef(self.data[self.var1].values, self.data[model_col].values)[0, 1]
            self.dia.add_sample(model_std, cc, label=model_col, **kwargs)

        self.fig.legend(
            self.dia.samplePoints,
            [p.get_label() for p in self.dia.samplePoints],
            numpoints=1,
            loc="upper right",
        )
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.fig.tight_layout()
        return self.ax
