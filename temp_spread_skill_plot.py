import numpy as np
import matplotlib.pyplot as plt
from src.monet_plots.plots.ensemble import SpreadSkillPlot

# Generate sample data
np.random.seed(42)
spread_values = np.random.rand(50) * 10 + 5
skill_values = spread_values + np.random.randn(50) * 2

# Create the plot
fig, ax = plt.subplots(figsize=(7, 7))
plotter = SpreadSkillPlot(spread=spread_values, skill=skill_values, fig=fig, ax=ax)
plotter.plot(alpha=0.7, color="blue", s=50)

# Save the plot
plt.tight_layout()
plt.savefig("/Users/barry/Documents/monet-plots/docs/examples/plots/spread_skill_plot.png")
plt.close(fig)

print("Spread-skill plot generated and saved.")
