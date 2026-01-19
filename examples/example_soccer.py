"""
Soccer
======

This example demonstrates Soccer.
"""

import pandas as pd
import matplotlib.pyplot as plt
from monet_plots.plots.soccer import SoccerPlot

# Create dummy data
df = pd.DataFrame({
    'obs': [10, 20, 30, 40],
    'mod': [12, 18, 35, 38],
    'site': ['Site A', 'Site B', 'Site C', 'Site D']
})

# Initialize Soccer Plot
plot = SoccerPlot(df, obs_col='obs', mod_col='mod', label_col='site',
                  goal={'bias': 20, 'error': 40},
                  criteria={'bias': 40, 'error': 60})

# Generate plot
plot.plot()
plot.save('soccer_example.png')
print("Soccer plot saved to soccer_example.png")
