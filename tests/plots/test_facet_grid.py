import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from monet_plots.plots.facet_grid import FacetGridPlot


@pytest.fixture
def clear_figures():
    """Clear all existing figures before and after a test."""
    plt.close("all")
    yield
    plt.close("all")


@pytest.fixture
def sample_data():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "x": np.random.rand(100),
            "y": np.random.rand(100),
            "row": np.random.choice(["a", "b"], 100),
            "col": np.random.choice(["c", "d"], 100),
        }
    )


def test_facet_grid_plot(clear_figures, sample_data):
    """Test FacetGridPlot."""
    plot = FacetGridPlot(data=sample_data, row="row", col="col")
    assert plot.grid is not None


def test_facets_wrap_and_map_grouped_frames(clear_figures):
    frame = pd.DataFrame(
        {
            "site": ["a", "c", "b", "a"],
            "model": ["control", "test", "control", "test"],
            "x": [0, 1, 2, 3],
            "y": [4, 5, 6, 7],
        }
    )
    plot = FacetGridPlot(
        frame,
        col="site",
        col_wrap=2,
        col_order=["c", "b", "a"],
        hue="model",
        hue_order=["control", "test"],
        palette={"control": "navy", "test": "orange"},
    )
    plot.map_dataframe(plt.scatter, "x", "y")
    plot.set_titles(col_template="Site {col_name}")
    plot.grid.add_legend()

    assert plot.grid.axs.shape == (2, 2)
    assert plot.grid.axes_dict["a"].get_title() == "Site a"
    assert not plot.grid.axs[-1, -1].get_visible()
    control, test = plot.grid.axes_dict["a"].collections
    np.testing.assert_array_equal(control.get_offsets(), [[0, 4]])
    np.testing.assert_array_equal(test.get_offsets(), [[3, 7]])
    labels = [text.get_text() for text in plot.fig.legends[0].get_texts()]
    assert labels == ["control", "test"]
