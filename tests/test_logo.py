import matplotlib.pyplot as plt
import numpy as np
import pytest
from monet_plots.plots.base import BasePlot
from monet_plots.plot_utils import get_logo_path
import os


def test_add_logo_default():
    plot = BasePlot()
    ab = plot.add_logo()
    assert ab is not None
    assert len(plot.ax.artists) == 1
    plt.close(plot.fig)


def test_add_logo_numpy():
    plot = BasePlot()
    logo_data = np.random.rand(10, 10, 3)
    ab = plot.add_logo(logo=logo_data)
    assert ab is not None
    assert len(plot.ax.artists) == 1
    plt.close(plot.fig)


def test_add_logo_positions():
    plot = BasePlot()
    positions = ["upper right", "upper left", "lower right", "lower left", "center"]
    for pos in positions:
        ab = plot.add_logo(loc=pos)
        assert ab is not None

    # Custom position
    ab = plot.add_logo(loc=(0.2, 0.2))
    assert ab is not None

    plt.close(plot.fig)


def test_get_logo_path():
    path = get_logo_path()
    assert os.path.exists(path)
    assert path.endswith("monet_logos.png")


@pytest.mark.skipif(
    os.environ.get("GITHUB_ACTIONS") == "true",
    reason="May not have internet access in CI",
)
def test_add_logo_url():
    # This might fail if no internet, so we should be careful
    plot = BasePlot()
    url = "https://raw.githubusercontent.com/noaa-oar-arl/monet-plots/main/src/monet_plots/assets/monet_logos.png"
    try:
        ab = plot.add_logo(logo=url)
        assert ab is not None
    except Exception as e:
        pytest.skip(f"Could not load logo from URL: {e}")
    finally:
        plt.close(plot.fig)
