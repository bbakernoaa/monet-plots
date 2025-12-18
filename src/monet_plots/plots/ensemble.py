import numpy as np
from .base import BasePlot


class SpreadSkillPlot(BasePlot):
    """Create a spread-skill plot to evaluate ensemble forecast reliability.

    This plot compares the standard deviation of the ensemble spread to the
    root mean squared error (RMSE) of the ensemble mean. A reliable ensemble
    should have a spread that is proportional to the forecast error.
    """

    def __init__(self, spread, skill, *args, **kwargs):
        """
        Initialize the plot with spread and skill data.

        Args:
            spread (array-like): The standard deviation of the ensemble forecast.
            skill (array-like): The root mean squared error of the ensemble mean.
        """
        super().__init__(*args, **kwargs)
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
