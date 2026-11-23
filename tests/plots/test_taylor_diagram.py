import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from monet_plots.plots.taylor_diagram import TaylorDiagramPlot


@pytest.fixture
def clear_figures():
    """Clear all existing figures before and after a test."""
    plt.close("all")
    yield
    plt.close("all")


@pytest.fixture
def sample_taylor_data():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "obs": np.linspace(0, 10, 100),
            "model": np.linspace(0, 10, 100) + np.random.rand(100),
        }
    )


def test_taylor_diagram_plot_creates_plot(clear_figures, sample_taylor_data):
    """Test that TaylorDiagramPlot creates a plot."""
    plot = TaylorDiagramPlot(df=sample_taylor_data, col1="obs", col2="model")
    dia = plot.plot()
    assert dia is not None


def test_taylor_diagram_model_sample_is_visible(clear_figures, sample_taylor_data):
    """Model sample should use a marker by default so single-point samples are visible."""
    plot = TaylorDiagramPlot(df=sample_taylor_data, col1="obs", col2="model")
    plot.plot()

    # samplePoints[0] is reference; samplePoints[1] is first model sample.
    assert len(plot.dia.samplePoints) >= 2
    assert plot.dia.samplePoints[1].get_marker() != "None"


def test_normalized_taylor_preserves_paired_error_statistics(clear_figures):
    data = pd.DataFrame(
        {
            "obs": [0.0, 2.0, 4.0, np.nan],
            "model": [1.0, np.nan, 7.0, 9.0],
            "scaled_obs": [0.0, 20.0, 40.0, np.nan],
            "scaled_model": [10.0, np.nan, 70.0, 90.0],
            "offset": [3.0, 5.0, 7.0, np.nan],
        }
    )
    plot = TaylorDiagramPlot(
        data,
        normalize=True,
        show_error_table=True,
        comparison_groups=[
            {
                "label": "First",
                "reference": "obs",
                "models": ["model", "offset"],
                "unit": "K",
            },
            {
                "label": "Second",
                "reference": "scaled_obs",
                "models": ["scaled_model"],
                "unit": "Pa",
            },
        ],
    )
    plot.plot()
    np.testing.assert_allclose(
        plot.dia.samples[1].get_data(), plot.dia.samples[3].get_data()
    )
    assert plot.dia.refstd == 1
    assert plot.error_table.iloc[0]["paired_count"] == 2
    assert plot.error_table.iloc[0]["bias"] == 2
    assert plot.error_table.iloc[0]["centered_rms"] == 1
    assert plot.error_table.iloc[0]["rmse"] == pytest.approx(np.sqrt(5))
    assert plot.error_table.iloc[1]["centered_rms"] == 0
    assert plot.error_table.iloc[1]["rmse"] == 3


def test_zero_reference_spread_retains_raw_errors(clear_figures):
    plot = TaylorDiagramPlot(
        pd.DataFrame({"obs": [2.0, 2.0], "model": [4.0, 4.0]}),
        normalize=True,
        show_error_table=True,
    )
    plot.plot()
    row = plot.error_table.iloc[0]
    assert len(plot.dia.samples) == 1
    assert row["paired_count"] == 2
    assert row["bias"] == row["rmse"] == 2
    assert pd.isna(row["normalized_rmse"])
    assert row["status"]


def test_wrapper_adds_outlying_model_to_existing_figure(clear_figures):
    frame = pd.DataFrame({"obs": [0, 1, 2], "model": [0, 1, 2]})
    first = TaylorDiagramPlot(frame)
    first.plot()
    sample = first.dia.samples[1]
    second = TaylorDiagramPlot(frame.assign(model=[20, 10, 0]), dia=first.dia)
    assert second.plot() is first.ax
    assert second.fig is first.fig
    assert first.dia.samples[1] is sample
    assert first.dia.smax > 10
    assert first.dia.samples[-1].get_xdata()[0] == pytest.approx(np.pi)


def test_journal_taylor_long_legend_stays_inside_page(clear_figures):
    name = "A very long operational model configuration label"
    plot = TaylorDiagramPlot(
        pd.DataFrame({"obs": [1, 2, 3], name: [0, 3, 6]}),
        var2=name,
        style="journal",
    )
    plot.plot()
    plot.fig.canvas.draw()
    legend = plot.ax.legend_.get_window_extent()
    assert plot.fig.bbox.contains(legend.x1, legend.y1)


def test_raw_taylor_table_matches_plotted_pairs_and_centered_rms(clear_figures):
    frame = pd.DataFrame(
        {"obs": [0, 2, 4, 6], "good": [1, 2, 7, 20], "sparse": [0, np.nan, 8, 9]}
    )
    plot = TaylorDiagramPlot(frame, var2=["good", "sparse"], show_error_table=True)
    plot.plot()
    row = plot.error_table.iloc[0]
    angle, radius = np.asarray(plot.dia.samples[1].get_data()).ravel()
    distance = np.hypot(
        radius * np.cos(angle) - plot.dia.refstd, radius * np.sin(angle)
    )
    assert row["paired_count"] == 3
    assert row["available_count"] == 4
    assert row["centered_rms"] == pytest.approx(distance)


def test_raw_taylor_table_reports_valid_pairs_when_common_geometry_is_unavailable(
    clear_figures,
):
    frame = pd.DataFrame(
        {"obs": [0, 2, 4], "first": [1, 3, np.nan], "second": [np.nan, 3, 5]}
    )
    plot = TaylorDiagramPlot(frame, var2=["first", "second"], show_error_table=True)
    plot.plot()

    assert plot.error_table["paired_count"].tolist() == [2, 2]
    assert plot.error_table["centered_rms"].tolist() == [0, 0]
    assert plot.error_table["status"].str.contains("common finite rows").all()


def test_shared_taylor_table_refreshes_from_either_wrapper(clear_figures):
    frame = pd.DataFrame({"obs": [0, 1, 2], "first": [0, 1, 2], "second": [2, 1, 0]})
    first = TaylorDiagramPlot(
        frame, var2="first", normalize=True, show_error_table=True
    )
    first.plot()
    second = TaylorDiagramPlot(
        frame, var2="second", dia=first.dia, normalize=True, show_error_table=True
    )
    second.plot()
    first.plot()

    assert sorted(first.error_table["model"]) == ["first", "second"]
    assert len(first.fig.axes) == 2
    assert len(first.dia.samples) == 3
