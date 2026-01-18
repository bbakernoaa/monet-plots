# tests/plots/test_fingerprint.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from monet_plots.plots.fingerprint import FingerprintPlot


def test_fingerprint_plot():
    dates = pd.date_range("2023-01-01", periods=1000, freq="h")
    df = pd.DataFrame({
        'time': dates,
        'val': np.random.rand(1000)
    })

    plot = FingerprintPlot(df, val_col='val', x_scale='hour', y_scale='dayofyear')
    ax = plot.plot()
    assert isinstance(ax, plt.Axes)
    assert ax.get_xlabel() == "Hour"
    assert ax.get_ylabel() == "Dayofyear"
    plt.close(plot.fig)


def test_fingerprint_plot_custom():
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    df = pd.DataFrame({
        'time': dates,
        'val': np.random.rand(100),
        'site': ['A', 'B'] * 50
    })

    # Using 'site' as y_scale (it should pick it up from columns)
    plot = FingerprintPlot(df, val_col='val', x_scale='month', y_scale='site')
    ax = plot.plot()
    assert isinstance(ax, plt.Axes)
    plt.close(plot.fig)
