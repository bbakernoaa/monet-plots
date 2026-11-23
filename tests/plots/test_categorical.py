import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from monet_plots.plots.categorical import categorical_plot


@pytest.fixture
def clear_figures():
    """Clear all existing figures before and after a test."""
    plt.close("all")
    yield
    plt.close("all")


@pytest.fixture
def sample_da():
    """Create a sample DataArray for testing."""
    return xr.DataArray(
        np.random.rand(10, 3),
        dims=("instance", "category"),
        coords={
            "instance": np.arange(10),
            "category": ["A", "B", "C"],
            "site": ("instance", ["site1"] * 5 + ["site2"] * 5),
        },
    )


def test_categorical_plot_bar(clear_figures, sample_da):
    """Test that categorical_plot creates a bar plot."""
    fig, ax = categorical_plot(
        sample_da, x="category", y=sample_da.name or "value", kind="bar"
    )
    assert fig is not None
    assert ax is not None
    # Check if there are bars in the plot
    assert len(ax.flatten()[0].patches) > 0


def test_categorical_plot_violin(clear_figures, sample_da):
    """Test that categorical_plot creates a violin plot."""
    fig, ax = categorical_plot(
        sample_da, x="category", y=sample_da.name or "value", kind="violin"
    )
    assert fig is not None
    assert ax is not None
    # Check if there are violins in the plot
    assert len(ax.flatten()[0].collections) > 0


def test_categorical_plot_hue(clear_figures, sample_da):
    """Test that categorical_plot works with hue."""
    fig, ax = categorical_plot(
        sample_da, x="category", y=sample_da.name or "value", hue="site", kind="bar"
    )
    assert fig is not None
    assert ax is not None
    assert len(ax.flatten()[0].patches) > 0
    # Check for legend
    assert len(fig.legends) > 0


def test_categorical_plot_missing_x_y_kwargs(clear_figures, sample_da):
    """Test that categorical_plot raises ValueError if x or y is not provided."""
    with pytest.raises(ValueError):
        categorical_plot(sample_da, kind="bar")
    with pytest.raises(ValueError):
        categorical_plot(sample_da, x="category", kind="bar")
    with pytest.raises(ValueError):
        categorical_plot(sample_da, y=sample_da.name, kind="bar")


def test_grouped_bar_estimates_and_facet_order(clear_figures):
    data = xr.Dataset(
        {
            "category": ("sample", ["a", "a", "b", "b", "a", "a", "b", "b"]),
            "site": ("sample", ["north"] * 4 + ["south"] * 4),
            "member": ("sample", ["obs", "model"] * 4),
            "value": ("sample", [1, 3, 2, 4, 10, 30, 20, 40]),
        }
    )
    fig, axes = categorical_plot(
        data,
        x="category",
        y="value",
        col="site",
        col_order=["south", "north"],
        hue="member",
        hue_order=["obs", "model"],
        order=["b", "a"],
        errorbar=None,
        legend_labels=["Observation", "Model"],
    )
    assert axes.shape == (2,)
    for ax, expected in zip(axes.flat, [[20, 10, 40, 30], [2, 1, 4, 3]]):
        assert [bar.get_height() for bar in ax.patches] == expected
        assert [tick.get_text() for tick in ax.get_xticklabels()] == ["b", "a"]
    assert [text.get_text() for text in fig.legends[0].get_texts()] == [
        "Observation",
        "Model",
    ]


def test_bootstrap_bars_and_unnamed_chunked_data(clear_figures):
    data = xr.DataArray(
        [1.0, 3.0, 5.0, 7.0],
        dims="sample",
        coords={"category": ("sample", ["A"] * 4)},
    ).chunk({"sample": 2})
    fig, axes = categorical_plot(
        data,
        x="category",
        y="value",
        estimator=np.median,
        errorbar=("ci", 95),
        seed=13,
    )
    assert data.name is None
    assert axes[0, 0].patches[0].get_height() == 4.0
    assert len(axes[0, 0].lines) == 1
    fig.canvas.draw()


def test_weighted_native_scale_bars(clear_figures):
    data = xr.Dataset(
        {
            "day": ("sample", [2, 2, 5]),
            "value": ("sample", [10, 20, 30]),
            "weight": ("sample", [1, 3, 1]),
        }
    )
    _, axes = categorical_plot(
        data,
        x="day",
        y="value",
        weights="weight",
        native_scale=True,
        errorbar=None,
    )
    bars = axes[0, 0].patches
    assert [bar.get_height() for bar in bars] == [17.5, 30]
    np.testing.assert_allclose(
        [bar.get_x() + bar.get_width() / 2 for bar in bars], [2, 5]
    )


def test_categorical_matplotlib_compatibility_options(clear_figures):
    data = xr.Dataset(
        {
            "category": ("sample", ["a"] * 4),
            "group": ("sample", [0, 0, 10, 10]),
            "region": ("sample", ["north"] * 4),
            "value": ("sample", [1, 100, 4, 100]),
        }
    )
    fig, axes = categorical_plot(
        data,
        x="category",
        y="value",
        col="region",
        hue="group",
        hue_norm=(0, 20),
        palette="viridis",
        saturation=1,
        facet_kws={"ylim": (1, 1000), "despine": True},
        log_scale=True,
        errorbar=None,
    )
    ax = axes.flat[0]
    assert ax.get_yscale() == "log"
    assert ax.get_ylim() == (1, 1000)
    assert not ax.spines["top"].get_visible()
    np.testing.assert_allclose([bar.get_height() for bar in ax.patches], [10, 20])
    np.testing.assert_allclose(
        [bar.get_facecolor() for bar in ax.patches],
        plt.get_cmap("viridis")([0.0, 0.5]),
    )


def test_categorical_log_violin_uses_positive_measurements(clear_figures):
    data = xr.Dataset(
        {"category": ("sample", ["a"] * 4), "value": ("sample", [0, 1, 10, 100])}
    )
    _, axes = categorical_plot(
        data,
        x="value",
        y="category",
        kind="violin",
        log_scale=(10, False),
        cut=0,
        inner=None,
    )
    ax = axes.flat[0]
    assert ax.get_xscale() == "log"
    vertices = ax.collections[0].get_paths()[0].vertices
    np.testing.assert_allclose([vertices[:, 0].min(), vertices[:, 0].max()], [1, 100])


def test_categorical_log_native_positions(clear_figures):
    data = xr.Dataset({"category": ("sample", [1, 10]), "value": ("sample", [3, 4])})
    _, axes = categorical_plot(
        data,
        x="category",
        y="value",
        native_scale=True,
        log_scale=(True, False),
        errorbar=None,
    )
    ax = axes.flat[0]
    assert ax.get_xscale() == "log"
    assert ax.get_yscale() == "linear"
    centers = [
        np.sqrt(bar.get_x() * (bar.get_x() + bar.get_width())) for bar in ax.patches
    ]
    np.testing.assert_allclose(centers, [1, 10])
