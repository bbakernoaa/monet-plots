# tests/plots/test_conditional_quantile.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from monet_plots.plots.conditional_quantile import ConditionalQuantilePlot


def test_conditional_quantile_plot():
    df = pd.DataFrame({
        'obs': np.linspace(0, 100, 100),
        'mod': np.linspace(0, 100, 100) + np.random.randn(100) * 10
    })

    plot = ConditionalQuantilePlot(df, obs_col='obs', mod_col='mod', bins=5)
    ax = plot.plot(show_points=True)
    assert isinstance(ax, plt.Axes)
    assert ax.get_xlabel() == "Observed: obs"
    assert ax.get_ylabel() == "Modeled: mod"
    plt.close(plot.fig)
