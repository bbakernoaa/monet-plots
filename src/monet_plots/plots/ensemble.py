from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from .base import BasePlot

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure


class SpreadSkillPlot(BasePlot):
    """Create a spread-skill plot to evaluate ensemble forecast reliability.

    This plot compares the standard deviation of the ensemble spread to the
    root mean squared error (RMSE) of the ensemble mean. A reliable ensemble
    should have a spread that is proportional to the forecast error.
    """

    def __init__(
        self,
        spread: Any,
        skill: Any,
        fig: Optional[matplotlib.figure.Figure] = None,
        ax: Optional[matplotlib.axes.Axes] = None,
        **kwargs: Any,
    ):
        """
        Initialize the plot with spread and skill data.

        Args:
            spread (array-like): The standard deviation of the ensemble forecast.
            skill (array-like): The root mean squared error of the ensemble mean.
            fig (matplotlib.figure.Figure, optional): Existing figure.
            ax (matplotlib.axes.Axes, optional): Existing axes.
            **kwargs: Additional keyword arguments for BasePlot.
        """
        super().__init__(fig=fig, ax=ax, **kwargs)
        self.spread = np.asarray(spread)
        self.skill = np.asarray(skill)

    def plot(self, **kwargs):
        """Generate the spread-skill plot.

        Additional keyword arguments are passed to the scatter plot.
        """
        # Plot the spread-skill pairs
        self.ax.scatter(self.spread, self.skill, **kwargs)

        # Add a 1:1 reference line
        max_val = max(np.max(self.spread), np.max(self.skill))
        self.ax.plot([0, max_val], [0, max_val], "k--")

        # Set labels and title
        self.ax.set_xlabel("Ensemble Spread (Standard Deviation)")
        self.ax.set_ylabel("Ensemble Error (RMSE)")
        self.ax.set_title("Spread-Skill Plot")

        # Ensure aspect ratio is equal
        self.ax.set_aspect("equal", "box")

        return self.ax

    def hvplot(self, **kwargs):
        """Generate an interactive spread-skill plot using hvPlot."""
        import hvplot.pandas  # noqa: F401
        import pandas as pd
        import holoviews as hv

        df = pd.DataFrame({"spread": self.spread, "skill": self.skill})

        plot_kwargs = {
            "x": "spread",
            "y": "skill",
            "kind": "scatter",
            "xlabel": "Ensemble Spread (Standard Deviation)",
            "ylabel": "Ensemble Error (RMSE)",
            "title": "Spread-Skill Plot",
        }
        plot_kwargs.update(kwargs)

        p = df.hvplot(**plot_kwargs)

        max_val = max(df["spread"].max(), df["skill"].max())
        one_to_one = hv.Curve([(0, 0), (max_val, max_val)]).opts(
            color="black", alpha=0.5, line_dash="dashed"
        )

        return one_to_one * p
