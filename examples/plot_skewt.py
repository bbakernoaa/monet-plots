"""Native Skew-T diagram with wind and parcel traces."""

import matplotlib.pyplot as plt
import numpy as np

from monet_plots import SkewTPlot

pressure = np.array([1000, 925, 850, 700, 500, 300, 200, 100])
temperature = np.array([24, 18, 13, 2, -18, -42, -54, -68])
dewpoint = np.array([19, 15, 8, -5, -26, -49, -61, -75])
u = np.array([1, 3, 5, 8, 13, 18, 22, 27])
v = np.array([2, 4, 6, 9, 12, 15, 17, 20])

fig, (left, right) = plt.subplots(
    1, 2, figsize=(12, 7), subplot_kw={"projection": "monet_skewt"}
)
SkewTPlot(ax=left).plot(
    pressure, temperature, dewpoint, u=u, v=v, label="Station A", parcel=True
)
comparison = SkewTPlot(ax=right)
comparison.plot(pressure, temperature + 1, dewpoint, label="Model")
comparison.plot(pressure, temperature, dewpoint, label="Observed", color="black")
fig.tight_layout()
fig.savefig("skewt.png", dpi=150)
