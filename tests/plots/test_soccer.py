# tests/plots/test_soccer.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from monet_plots.plots.soccer import SoccerPlot


def test_soccer_plot_calculation():
    df = pd.DataFrame(
        {"obs": [10, 20, 30], "mod": [12, 18, 35], "label": ["A", "B", "C"]}
    )

    # Test fractional metric
    plot = SoccerPlot(df, obs_col="obs", mod_col="mod", metric="fractional")
    assert len(plot.bias_data) == 3
    assert len(plot.error_data) == 3
    # MFB = 2 * (12-10)/(12+10) = 2 * 2 / 22 = 4/22 approx 0.1818 -> 18.18%
    bias0 = (
        plot.bias_data.iloc[0] if hasattr(plot.bias_data, "iloc") else plot.bias_data[0]
    )
    assert np.isclose(bias0, 200.0 * 2 / 22)

    # Test plotting
    ax = plot.plot()
    assert isinstance(ax, plt.Axes)
    plt.close(plot.fig)


def test_soccer_plot_precalculated():
    df = pd.DataFrame({"my_bias": [5, -10], "my_error": [20, 30]})
    plot = SoccerPlot(df, bias_col="my_bias", error_col="my_error")
    assert (plot.bias_data == df["my_bias"]).all()
    ax = plot.plot()
    assert isinstance(ax, plt.Axes)
    plt.close(plot.fig)


def test_soccer_plot_labels():
    df = pd.DataFrame({"obs": [10, 20], "mod": [12, 18], "site": ["Site 1", "Site 2"]})
    plot = SoccerPlot(df, obs_col="obs", mod_col="mod", label_col="site")
    ax = plot.plot()
    # Check if annotations are added (hard to check exactly but we can check if it runs)
    assert isinstance(ax, plt.Axes)
    plt.close(plot.fig)
