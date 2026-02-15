"""
Relative Economic Value Plot
============================

**What it's for:**
The Relative Economic Value plot (also known as the Richardson plot) assesses the
practical utility of a forecast system for a user who must make a decision based
on a cost-loss ratio.

**When to use:**
Use this to move beyond purely statistical metrics and communicate the "value" of
a model to stakeholders. It helps answer whether using the model's forecasts
is better than simply relying on climatology or always taking a protective action.

**How to read:**
*   **X-axis:** The Cost/Loss (C/L) ratio of the user. This represents the cost of
    taking action divided by the loss incurred if an event happens and no action was taken.
*   **Y-axis:** Relative Economic Value (V).
*   **The Curve:** Shows the value for each possible cost-loss ratio.
*   **Interpretation:** A value of 1 represents a perfect forecast. A value of 0
    means the forecast is no more valuable than simply using climatology. The
    highest point on the curve indicates the cost-loss ratio for which the
    forecast system is most beneficial.
"""

import pandas as pd
import matplotlib.pyplot as plt
from monet_plots.plots.rev import RelativeEconomicValuePlot

# 1. Prepare sample data with contingency table counts
# These counts represent the performance of a forecast system over many events
data = {
    "hits": [100],
    "misses": [20],
    "fa": [30],  # False Alarms
    "cn": [850],  # Correct Negatives
}
df = pd.DataFrame(data)

# Define the columns for hits, misses, false alarms, and correct negatives
contingency_cols = ["hits", "misses", "fa", "cn"]

# 2. Initialize and create the plot
plot = RelativeEconomicValuePlot(df, counts_cols=contingency_cols, figsize=(10, 7))
plot.plot(color="blue", linewidth=2)

# 3. Add titles and labels
plot.ax.set_title("Relative Economic Value of a Forecast System")
plot.ax.legend(loc="lower right")

plt.tight_layout()
plt.show()
