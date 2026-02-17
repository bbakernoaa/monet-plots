"""
Brier Decomposition Plot
========================

**What it's for:**
The Brier Score Decomposition plot visualizes the three components of the Brier Score:
Reliability, Resolution, and Uncertainty. The total Brier Score (BS) is calculated as:
BS = Reliability - Resolution + Uncertainty.

**When to use:**
Use this to gain a deeper understanding of why a probabilistic forecast system is performing
a certain way. It allows you to distinguish between a model that is poorly calibrated
(Reliability) and one that lacks the ability to distinguish different outcomes (Resolution).

**How to read:**
*   **Reliability (Lower is better):** Measures the weighted average of the squared
    differences between forecast probabilities and the relative frequencies of observed events.
*   **Resolution (Higher is better):** Measures how much the frequencies of events for
    specific forecast categories differ from the overall climatological frequency.
*   **Uncertainty (Fixed for a given set of observations):** Represents the inherent
    variability of the events being forecast.
*   **Interpretation:** A perfect model would have a Reliability of 0 and a Resolution
    equal to the Uncertainty, resulting in a Brier Score of 0.
"""

import pandas as pd
import matplotlib.pyplot as plt
from monet_plots.plots.brier_decomposition import BrierScoreDecompositionPlot

data = pd.DataFrame(
    {
        "model": ["Model A", "Model B", "Model C"],
        "reliability": [0.02, 0.05, 0.01],
        "resolution": [0.15, 0.10, 0.18],
        "uncertainty": [0.25, 0.25, 0.25],
    }
)

plot = BrierScoreDecompositionPlot(figsize=(10, 7))
plot.plot(
    data,
    reliability_col="reliability",
    resolution_col="resolution",
    uncertainty_col="uncertainty",
    label_col="model",
    title="Brier Score Decomposition Comparison",
)

plt.tight_layout()
plt.show()
