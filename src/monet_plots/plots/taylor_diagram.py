import functools
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy import corrcoef
from .base import BasePlot
from ..plot_utils import to_dataframe
from typing import Any, Union, List
import mpl_toolkits.axisartist.floating_axes as FA
import mpl_toolkits.axisartist.grid_finder as GF
from matplotlib.projections import PolarAxes
import matplotlib.axes

# Define the color palette and decorator at the module level
colors = ["#DA70D6", "#228B22", "#FA8072", "#FF1493"]


def _sns_context(f):
    """Decorator to apply seaborn color palette to a function."""

    @functools.wraps(f)
    def inner(*args, **kwargs):
        with sns.color_palette(colors):
            return f(*args, **kwargs)

    return inner


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
        """
        super().__init__(*args, **kwargs)
        self.col1 = col1
        if isinstance(col2, str):
            self.col2 = [col2]
        else:
            self.col2 = col2
        required_cols = [self.col1] + self.col2
        self.df = to_dataframe(df).dropna(subset=required_cols)
        self.label1 = label1
        self.scale = scale
        self.sample_points = []
        self._ax = None  # To hold the floating axes

    def _setup_diagram(self, refstd: float) -> None:
        """Set up the Taylor diagram axes.
        Args:
            refstd (float): Reference standard deviation.
        """
        tr = PolarAxes.PolarTransform(apply_theta_transforms=False)
        rlocs = np.concatenate((np.arange(10) / 10.0, [0.95, 0.99]))
        tlocs = np.arccos(rlocs)
        gl1 = GF.FixedLocator(tlocs)
        tf1 = GF.DictFormatter(dict(zip(tlocs, map(str, rlocs))))
        smin = 0
        smax = self.scale * refstd
        ghelper = FA.GridHelperCurveLinear(
            tr,
            extremes=(0, np.pi / 2, smin, smax),
            grid_locator1=gl1,
            tick_formatter1=tf1,
        )

        # Remove the default axes and add the floating axes
        self.ax.remove()
        ax = FA.FloatingSubplot(self.fig, 111, grid_helper=ghelper)
        self.fig.add_subplot(ax)

        ax.axis["top"].set_axis_direction("bottom")
        ax.axis["top"].toggle(ticklabels=True, label=True)
        ax.axis["top"].major_ticklabels.set_axis_direction("top")
        ax.axis["top"].label.set_axis_direction("top")
        ax.axis["top"].label.set_text("Correlation")
        ax.axis["left"].set_axis_direction("bottom")
        ax.axis["left"].label.set_text("Standard deviation")
        ax.axis["right"].set_axis_direction("top")
        ax.axis["right"].toggle(ticklabels=True)
        ax.axis["right"].major_ticklabels.set_axis_direction("left")
        ax.axis["bottom"].set_visible(False)
        ax.grid(False)

        # Note: self.ax is the polar axes for plotting, self._ax is the container
        self._ax = ax
        self.ax = ax.get_aux_axes(tr)

        (line,) = self.ax.plot(
            [0], refstd, "r*", ls="", ms=14, label=self.label1, zorder=10
        )
        t = np.linspace(0, np.pi / 2)
        r = np.zeros_like(t) + refstd
        self.ax.plot(t, r, "k--", label="_")
        self.sample_points.append(line)

    def add_sample(self, stddev: float, corr: float, *args, **kwargs) -> None:
        """Add a sample point to the diagram.
        Args:
            stddev (float): Standard deviation of the sample.
            corr (float): Correlation of the sample.
        """
        (line,) = self.ax.plot(np.arccos(corr), stddev, *args, **kwargs)
        self.sample_points.append(line)

    def add_contours(self, levels: int = 5, **kwargs) -> None:
        """Add RMS contours to the diagram.
        Args:
            levels (int): Number of contour levels.
        """
        refstd = self.df[self.col1].std()
        smin = 0
        smax = self.scale * refstd
        rs, ts = np.meshgrid(np.linspace(smin, smax), np.linspace(0, np.pi / 2))
        rms = np.sqrt(refstd**2 + rs**2 - 2 * refstd * rs * np.cos(ts))
        contours = self.ax.contour(ts, rs, rms, levels, **kwargs)
        plt.clabel(contours, inline=1, fontsize=10)

    @_sns_context
    def plot(self, **kwargs) -> matplotlib.axes.Axes:
        """Generate the Taylor diagram.
        Returns:
            matplotlib.axes.Axes: The axes object containing the plot.
        """
        refstd = self.df[self.col1].std()
        self._setup_diagram(refstd)
        for model_col in self.col2:
            model_std = self.df[model_col].std()
            cc = corrcoef(self.df[self.col1].values, self.df[model_col].values)[0, 1]
            self.add_sample(model_std, cc, label=model_col, **kwargs)
        self.add_contours(colors="0.5")
        self.fig.legend(
            self.sample_points,
            [p.get_label() for p in self.sample_points],
            numpoints=1,
            loc="upper right",
        )
        self.fig.tight_layout()
        return self._ax
