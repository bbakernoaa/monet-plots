import numpy as np
import matplotlib.colors as mcolors
from monet_plots.colorbars import (
    get_linear_scale,
    get_diverging_scale,
    get_discrete_scale,
    get_log_scale,
)


def test_get_linear_scale():
    data = np.array([0, 1, 2, 3, 4, 5])
    cmap, norm = get_linear_scale(data, vmin=1, vmax=4)
    assert isinstance(norm, mcolors.Normalize)
    assert norm.vmin == 1
    assert norm.vmax == 4

    # Percentiles
    cmap, norm = get_linear_scale(data, p_min=10, p_max=90)
    assert norm.vmin == np.percentile(data, 10)
    assert norm.vmax == np.percentile(data, 90)


def test_get_diverging_scale():
    data = np.array([-10, 0, 5])
    cmap, norm = get_diverging_scale(data, center=0)
    assert norm.vmin == -10
    assert norm.vmax == 10

    cmap, norm = get_diverging_scale(data, center=2, span=5)
    assert norm.vmin == -3
    assert norm.vmax == 7

    # p_span
    data = np.array([-10, -5, 0, 5, 10, 20])
    cmap, norm = get_diverging_scale(data, center=0, p_span=50)
    # diffs are [10, 5, 0, 5, 10, 20], median (50th percentile) is around 7.5 (interpolated)
    expected_span = np.percentile(np.abs(data), 50)
    assert norm.vmin == -expected_span
    assert norm.vmax == expected_span


def test_get_discrete_scale():
    data = np.array([0, 10.5, 20.1, 35.6])
    cmap, norm = get_discrete_scale(data, n_levels=5)
    assert isinstance(norm, mcolors.BoundaryNorm)
    # MaxNLocator should give "nice" numbers
    assert norm.boundaries[0] <= 0
    assert norm.boundaries[-1] >= 35.6
    # Check if they are "nice" (multiples of 1, 2, 2.5, 5, 10)
    diffs = np.diff(norm.boundaries)
    assert np.all(
        np.isclose(diffs, diffs[0])
    )  # Should be equal intervals for MaxNLocator


def test_get_log_scale():
    data = np.array([0.1, 1, 10, 100])
    cmap, norm = get_log_scale(data)
    assert isinstance(norm, mcolors.LogNorm)
    assert norm.vmin == 0.1
    assert norm.vmax == 100

    # Handling non-positive data
    data = np.array([-1, 0, 1, 10])
    cmap, norm = get_log_scale(data)
    assert norm.vmin > 0
    assert norm.vmax == 10
