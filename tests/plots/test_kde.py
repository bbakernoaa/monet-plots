import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.stats import gaussian_kde

from monet_plots.plots.kde import KDEPlot


def test_bivariate_kde_renders_joint_density():
    rng = np.random.default_rng(17)
    data = pd.DataFrame(rng.normal(size=(100, 2)), columns=["model", "observed"])
    plot = KDEPlot(data, x="model", y="observed")
    try:
        ax = plot.plot(levels=[0.25, 0.5, 0.75])
        assert not ax.lines
        assert ax.collections
        assert ax.get_ylabel() == "observed"
    finally:
        plot.close()


def test_weighted_kde_matches_scipy():
    data = pd.DataFrame({"model": [0, 1, 2, 3], "weight": [1, 2, 3, 5]})
    plot = KDEPlot(data, "model")
    try:
        line = plot.plot(weights="weight", bw_adjust=0.8, cut=0).lines[0]
        kde = gaussian_kde(data.model, weights=data.weight)
        kde.set_bandwidth(kde.factor * 0.8)
        np.testing.assert_allclose(line.get_ydata(), kde(line.get_xdata()))
    finally:
        plot.close()


def test_grouped_kde_normalizes_by_sample_mass():
    data = pd.DataFrame(
        {
            "model": [0, 1, 2, 3, 4],
            "group": pd.Categorical(["A", "A", "A", "B", "B"]),
        }
    )
    plot = KDEPlot(data, "model")
    try:
        ax = plot.plot(hue="group", common_grid=True, cut=8)
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), ax.lines[1].get_xdata())
        masses = [trapezoid(line.get_ydata(), line.get_xdata()) for line in ax.lines]
        np.testing.assert_allclose(masses, [0.6, 0.4], atol=1e-5)
    finally:
        plot.close()


def test_kde_handles_nullable_hue_values_and_rgb_palette():
    data = pd.DataFrame(
        {
            "model": [0, 1, 2, 3, 100],
            "group": pd.Series(["A", "A", "B", "B", None], dtype="string"),
        }
    )
    plot = KDEPlot(data, "model")
    try:
        ax = plot.plot(hue="group", palette=[1, 0, 0], cut=0)
        assert [line.get_label() for line in ax.lines] == ["A", "B"]
        assert ax.lines[0].get_xdata().max() == 1
        assert ax.lines[1].get_color() == [1, 0, 0]
    finally:
        plot.close()
