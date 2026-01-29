import cartopy.crs as ccrs
from monet_plots.plots.spatial import SpatialPlot
import matplotlib.pyplot as plt

def test_style():
    custom_style = {"linewidth": 2.5, "edgecolor": "red"}
    plot = SpatialPlot(states=custom_style)
    for coll in plot.ax.collections:
        print(f"Type: {type(coll)}, LW: {coll.get_linewidth()}")

test_style()
