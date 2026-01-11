"""
Performance Diagram
===================

This example demonstrates the Performance Diagram, which is used to evaluate
the performance of a categorical forecast system. It plots the Success Ratio
against the Probability of Detection.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from monet_plots.plots.performance_diagram import PerformanceDiagram

# 1. Create synthetic categorical forecast data
# This represents a series of forecasts and their corresponding observations
# for a specific event (e.g., fog).
# H (Hits): Forecast said yes, Observation was yes.
# M (Misses): Forecast said no, Observation was yes.
# F (False Alarms): Forecast said yes, Observation was no.
# C (Correct Negatives): Forecast said no, Observation was no.
np.random.seed(42)
n_forecasts = 5
data = []
for i in range(n_forecasts):
    hits = np.random.randint(50, 100)
    misses = np.random.randint(10, 30)
    false_alarms = np.random.randint(20, 50)
    correct_negatives = np.random.randint(200, 400)
    data.append([f"Model_{i+1}", hits, misses, false_alarms, correct_negatives])

df = pd.DataFrame(data, columns=["model", "hits", "misses", "false_alarms", "correct_negatives"])


# 2. Create and display the plot
fig, ax = plt.subplots(figsize=(8, 8))
plot = PerformanceDiagram(ax=ax, data=df)
plot.plot(
    label_col="model",
    title="Performance Diagram for Multiple Forecast Models",
)
plt.show()
