"""
Performance Diagram
===================

This example demonstrates how to create a Performance Diagram.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from monet_plots.plots.performance_diagram import PerformanceDiagramPlot

# 1. Prepare sample data with pre-calculated metrics
# Each row represents a different forecast threshold or scenario
data = {
    'success_ratio': [0.2, 0.4, 0.6, 0.8, 0.9],
    'pod': [0.3, 0.6, 0.7, 0.85, 0.92],
    'model': ['Model A'] * 5 # For a single model
}
df = pd.DataFrame(data)

# 2. Initialize and create the plot
plot = PerformanceDiagramPlot(figsize=(8, 8))
plot.plot(df, x_col='success_ratio', y_col='pod', markersize=8, color='blue', label='Forecast System')

# 3. Add titles and labels (optional, but good practice)
plot.ax.set_title("Performance Diagram for a Forecast System")
plot.ax.legend(loc='lower right') # Add legend for the plotted points

plt.tight_layout()
plt.show()
