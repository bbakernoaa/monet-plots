# tests/plots/test_polar.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from monet_plots.plots.polar import BivariatePolarPlot


def test_polar_plot():
    df = pd.DataFrame(
        {
            "ws": np.random.rand(100) * 10,
            "wd": np.random.rand(100) * 360,
            "conc": np.random.rand(100) * 50,
        }
    )

    plot = BivariatePolarPlot(df, ws_col="ws", wd_col="wd", val_col="conc")
    ax = plot.plot()
    assert isinstance(ax, plt.Axes)
    # Check if it's polar
    assert ax.name == "polar"
    plt.close(plot.fig)
