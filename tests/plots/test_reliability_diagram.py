import numpy as np
import pandas as pd
import pytest
import xarray as xr
from dask.callbacks import Callback

from monet_plots.plots.reliability_diagram import ReliabilityDiagramPlot


def test_reliability_diagram_plot_pre_binned():
    """Test the ReliabilityDiagramPlot plot method with pre-binned data."""
    plot = ReliabilityDiagramPlot()
    data = pd.DataFrame({"prob": np.linspace(0, 1, 11), "freq": np.linspace(0, 1, 11)})
    plot.plot(data)
    assert plot.ax is not None


def test_reliability_diagram_plot_raw_data():
    """Test the ReliabilityDiagramPlot plot method with raw data."""
    plot = ReliabilityDiagramPlot()
    data = pd.DataFrame(
        {"forecasts": np.random.rand(100), "observations": np.random.randint(0, 2, 100)}
    )
    plot.plot(data, forecasts_col="forecasts", observations_col="observations")
    assert plot.ax is not None


def test_reliability_diagram_plot_show_hist():
    """Test the ReliabilityDiagramPlot plot method with show_hist=True."""
    plot = ReliabilityDiagramPlot()
    data = pd.DataFrame(
        {
            "prob": np.linspace(0, 1, 11),
            "freq": np.linspace(0, 1, 11),
            "count": np.random.randint(1, 100, 11),
        }
    )
    plot.plot(data, show_hist=True)
    assert plot.ax is not None


def test_empty_bins_are_gaps_with_correct_skill_boundary():
    plot = ReliabilityDiagramPlot()
    try:
        data = pd.DataFrame(
            {"prob": [0.1, 0.5, 0.9], "freq": [0.2, 0, 0.8], "count": [5, 0, 5]}
        )
        plot.plot(data, climatology=0.4)
        assert np.isnan(plot.ax.lines[-1].get_ydata()[1])
        boundary = next(
            line for line in plot.ax.lines if line.get_label() == "No skill"
        )
        np.testing.assert_allclose(boundary.get_ydata(), [0.2, 0.7])
    finally:
        plot.close()


def test_raw_grouped_reliability_bins_each_model_independently():
    data = pd.DataFrame(
        {
            "model": ["A", "A", "B", "B", "B"],
            "probability": [0.1, 0.1, 0.1, 0.9, np.nan],
            "event": [0, 0, 1, 1, 0],
        }
    )
    plot = ReliabilityDiagramPlot()
    try:
        plot.plot(
            data,
            forecasts_col="probability",
            observations_col="event",
            label_col="model",
            n_bins=2,
            show_hist=True,
        )
        lines = {line.get_label(): line for line in plot.ax.lines}
        np.testing.assert_allclose(lines["A"].get_ydata(), [0, np.nan])
        np.testing.assert_allclose(lines["B"].get_ydata(), [1, 1])
        np.testing.assert_allclose(lines["Climatology"].get_ydata(), [0.5, 0.5])
        assert sorted(bar.get_height() for bar in plot.fig.axes[-1].patches) == [1, 3]
    finally:
        plot.close()


@pytest.mark.parametrize("label_col", [None, "model"])
@pytest.mark.parametrize("backend", ["xarray", "dataframe"])
def test_raw_dask_reliability_matches_eager_without_full_frame(
    monkeypatch, label_col, backend
):
    data = pd.DataFrame(
        {
            "forecast": [0, 0.1, 0.25, 0.25, 0.5, 0.75, 1, 0.8, -0.1, 1.1, np.nan, 0.9],
            "obs": [0, 1, 1, 0, 1, 0, 1, 2, 1, 1, 1, 1],
            "model": ["z", "a", "a", "z", "z", "a", "a", "a", "a", "z", "a", None],
        }
    )

    def reject_raw_frame(*args, **kwargs):
        pytest.fail("Raw Dask input became a pandas frame")

    if backend == "xarray":
        lazy_data = xr.Dataset.from_dataframe(data).chunk({"index": 3})
        monkeypatch.setattr(xr.Dataset, "to_dataframe", reject_raw_frame)
        monkeypatch.setattr(xr.Dataset, "compute", reject_raw_frame)
    else:
        dd = pytest.importorskip("dask.dataframe")
        lazy_data = dd.from_pandas(data, npartitions=4)
        monkeypatch.setattr(dd.DataFrame, "compute", reject_raw_frame)

    eager, lazy = ReliabilityDiagramPlot(), ReliabilityDiagramPlot()
    options = dict(
        forecasts_col="forecast",
        observations_col="obs",
        label_col=label_col,
        n_bins=4,
        show_hist=True,
    )
    runs = []
    try:
        eager.plot(data, **options)
        with Callback(start=lambda graph: runs.append(len(graph))):
            lazy.plot(lazy_data, **options)
        assert len(runs) == 1
        assert [line.get_label() for line in lazy.ax.lines] == [
            line.get_label() for line in eager.ax.lines
        ]
        for actual, expected in zip(lazy.ax.lines, eager.ax.lines, strict=True):
            np.testing.assert_allclose(actual.get_xydata(), expected.get_xydata())
        for actual, expected in zip(
            lazy.fig.axes[-1].patches, eager.fig.axes[-1].patches, strict=True
        ):
            np.testing.assert_allclose(
                [actual.get_x(), actual.get_height()],
                [expected.get_x(), expected.get_height()],
            )
    finally:
        eager.close()
        lazy.close()


def test_raw_dask_reliability_broadcasts_outcomes_over_models():
    data = xr.Dataset(
        {
            "forecast": (("model", "case"), [[0.0, 0.5, 1.0], [1.0, 0.5, 0.0]]),
            "obs": ("case", [0, 1, 1]),
        },
        coords={"model": ["first", "second"]},
    ).chunk({"case": 1, "model": 1})
    plot = ReliabilityDiagramPlot()
    try:
        plot.plot(
            data,
            forecasts_col="forecast",
            observations_col="obs",
            label_col="model",
            n_bins=2,
            climatology=0.1,
        )
        lines = {line.get_label(): line for line in plot.ax.lines}
        np.testing.assert_allclose(
            lines["first"].get_xydata(), [[0.0, 0.0], [0.75, 1.0]]
        )
        np.testing.assert_allclose(
            lines["second"].get_xydata(), [[0.0, 1.0], [0.75, 0.5]]
        )
        np.testing.assert_allclose(lines["Climatology"].get_ydata(), [0.1, 0.1])
    finally:
        plot.close()


def test_raw_dask_reliability_rejects_invalid_pairs():
    data = xr.Dataset({"forecast": ("case", [-1, 2]), "obs": ("case", [0, 3])}).chunk()
    plot = ReliabilityDiagramPlot()
    try:
        with pytest.raises(ValueError, match="No valid forecast probabilities"):
            plot.plot(data, forecasts_col="forecast", observations_col="obs")
    finally:
        plot.close()


def test_raw_dask_reliability_counts_match_full_dataset_broadcast():
    data = xr.Dataset(
        {"forecast": ("case", [0.0, 0.5, 1.0]), "obs": ("case", [0, 1, 1])},
        coords={"replica": [0, 1, 2]},
    )
    eager, lazy = ReliabilityDiagramPlot(), ReliabilityDiagramPlot()
    try:
        options = dict(
            forecasts_col="forecast", observations_col="obs", n_bins=2, show_hist=True
        )
        eager.plot(data, **options)
        lazy.plot(data.chunk({"case": 1}), **options)
        np.testing.assert_allclose(
            [bar.get_height() for bar in lazy.fig.axes[-1].patches],
            [bar.get_height() for bar in eager.fig.axes[-1].patches],
        )
    finally:
        eager.close()
        lazy.close()
