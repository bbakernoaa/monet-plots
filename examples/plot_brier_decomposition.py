"""
Brier Decomposition Plot
========================

This example demonstrates how to create a Brier Decomposition plot.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from monet_plots.plots.brier_decomposition import BrierScoreDecompositionPlot

data = pd.DataFrame({
    'model': ['Model A', 'Model B', 'Model C'],
    'reliability': [0.02, 0.05, 0.01],
    'resolution': [0.15, 0.10, 0.18],
    'uncertainty': [0.25, 0.25, 0.25]
})

plot = BrierScoreDecompositionPlot(figsize=(10, 7))
plot.plot(
    data,
    reliability_col='reliability',
    resolution_col='resolution',
    uncertainty_col='uncertainty',
    label_col='model',
    title="Brier Score Decomposition Comparison"
)

plt.tight_layout()
plt.show()
