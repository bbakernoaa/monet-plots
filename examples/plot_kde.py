"""
Kernel Density Estimate (KDE) Plot
==================================

**What it's for:**
A KDE plot visualizes the probability density of one or more variables. It provides a
smoothed version of a histogram, making it easier to identify the underlying
distribution of the data.

**When to use:**
Use this to visualize the distribution of model errors, concentrations, or any
continuous variable. In 2D (as shown in this example), it is excellent for
visualizing the joint distribution and correlation between two variables without
the clutter of a scatter plot.

**How to read:**
*   **Axes:** Represent the variables being analyzed.
*   **Contour/Color:** Higher intensity or specific contour levels indicate regions with
    a higher density of data points (i.e., where values are more likely to occur).
*   **Interpretation:** The peak of the density represents the most frequent value (mode).
    In 2D, the shape of the density (e.g., an elongated ellipse) indicates the
    correlation between the two variables.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from monet_plots.plots.kde import KDEPlot

# 1. Prepare sample data
np.random.seed(42)  # for reproducibility
mean = [0, 0]
cov = [[1, 0.8], [0.8, 1]]  # Correlated variables
data = np.random.multivariate_normal(mean, cov, 1000)
df = pd.DataFrame(data, columns=["Variable A", "Variable B"])

# 2. Initialize and create the plot
plot = KDEPlot(
    df=df,
    x="Variable A",
    y="Variable B",
    title="2D Kernel Density Estimate of Correlated Variables",
    figsize=(8, 7),
)
plot.plot()  # Default KDE plot

# 3. Add titles and labels
plot.ax.set_xlabel("Variable A")
plot.ax.set_ylabel("Variable B")

plt.tight_layout()
plt.show()
