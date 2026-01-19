"""
Rev
===

This example demonstrates how to create a Rev.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from monet_plots.plots.rev import RelativeEconomicValuePlot

# 1. Prepare sample data with contingency table counts
# These counts represent the performance of a forecast system over many events
data = {
    'hits': [100],
    'misses': [20],
    'fa': [30], # False Alarms
    'cn': [850] # Correct Negatives
}
df = pd.DataFrame(data)

# Define the columns for hits, misses, false alarms, and correct negatives
contingency_cols = ['hits', 'misses', 'fa', 'cn']

# 2. Initialize and create the plot
plot = RelativeEconomicValuePlot(figsize=(10, 7))
plot.plot(
    df,
    counts_cols=contingency_cols,
    color='blue',
    linewidth=2
)

# 3. Add titles and labels
plot.ax.set_title("Relative Economic Value of a Forecast System")
plot.ax.legend(loc='lower right')

plt.tight_layout()
plt.show()
