"""
Meteogram
=========

This example demonstrates how to create a Meteogram.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from monet_plots.plots.meteogram import Meteogram

# 1. Prepare sample data
np.random.seed(42) # for reproducibility
dates = pd.date_range('2023-01-01 00:00', periods=24, freq='h')

# Simulate temperature with a diurnal cycle
temperature = 20 + 5 * np.sin(np.linspace(0, 2 * np.pi, 24)) + np.random.normal(0, 0.5, 24)
# Simulate humidity
humidity = 70 - 10 * np.cos(np.linspace(0, 2 * np.pi, 24)) + np.random.normal(0, 1, 24)
# Simulate pressure
pressure = 1012 + 3 * np.sin(np.linspace(0, 2 * np.pi, 24) + np.pi/4) + np.random.normal(0, 0.2, 24)

df = pd.DataFrame({
    'Temperature': temperature,
    'Humidity': humidity,
    'Pressure': pressure
}, index=dates)

# 2. Initialize and create the plot
plot = Meteogram(df=df, variables=['Temperature', 'Humidity', 'Pressure'], figsize=(12, 9))
plot.plot(linewidth=1.5, marker='o', markersize=3) # Plot with lines and markers

# Add an overall title to the figure
plot.fig.suptitle("Synthetic Meteogram for a 24-hour Period", fontsize=16)

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
plt.show()
