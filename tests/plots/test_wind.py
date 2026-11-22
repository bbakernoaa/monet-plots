import matplotlib.pyplot as plt
import numpy as np
import pytest

from monet_plots.plots.wind_barbs import WindBarbsPlot
from monet_plots.plots.wind_quiver import WindQuiverPlot


class GridObj:
    def __init__(self):
        self.variables = {
            "LAT": np.random.rand(1, 1, 10, 10),
            "LON": np.random.rand(1, 1, 10, 10),
        }


class Basemap:
    def __call__(self, lon, lat):
        return lon, lat

    def barbs(self, *args, **kwargs):
        pass

    def quiver(self, *args, **kwargs):
        kwargs.pop("ax", None)
        return plt.quiver(*args, **kwargs)


@pytest.fixture
def clear_figures():
    """Clear all existing figures before and after a test."""
    plt.close("all")
    yield
    plt.close("all")


@pytest.fixture
def sample_data():
    """Create a sample wind speed and direction data for testing."""
    return np.random.rand(10, 10), np.random.rand(10, 10)


def test_wind_barbs_plot(clear_figures, sample_data):
    """Test WindBarbsPlot."""
    ws, wdir = sample_data
    plot = WindBarbsPlot(ws=ws, wdir=wdir, gridobj=GridObj())
    ax = plot.plot()
    assert ax is not None


def test_wind_quiver_plot(clear_figures, sample_data):
    """Test WindQuiverPlot."""
    ws, wdir = sample_data
    plot = WindQuiverPlot(ws=ws, wdir=wdir, gridobj=GridObj())
    quiv = plot.plot()
    assert quiv is not None


def test_wind_defaults_keep_small_grid_and_allow_explicit_thinning(
    clear_figures, sample_data
):
    ws, wdir = sample_data
    arrows = WindQuiverPlot(ws, wdir, GridObj(), coastlines=False)
    assert arrows.plot().U.size == 100
    barbs = WindBarbsPlot(ws, wdir, GridObj(), coastlines=False)
    barbs.plot(skip=2)
    assert barbs.ax.collections[-1].u.size == 25
