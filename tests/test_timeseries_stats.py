import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from monet_plots.plots.timeseries import TimeSeriesStatsPlot


@pytest.fixture
def sample_ts_data():
    """Create sample data for testing."""
    dates = pd.date_range("2023-01-01", periods=20, freq="h")
    obs = np.sin(np.linspace(0, 2 * np.pi, 20))
    mod = obs + 0.1
    return dates, obs, mod


def test_timeseries_stats_pandas(sample_ts_data):
    """Test TimeSeriesStatsPlot with pandas DataFrame."""
    dates, obs, mod = sample_ts_data
    df = pd.DataFrame({"time": dates, "obs": obs, "mod": mod})

    plot = TimeSeriesStatsPlot(df, col1="obs", col2="mod", x="time")
    assert plot.x == "time"

    ax = plot.plot(stat="bias", freq="5h")
    assert ax is not None

    # Check if lines were plotted
    assert len(ax.get_lines()) == 1
    plt.close(plot.fig)


def test_timeseries_stats_xarray_lazy(sample_ts_data):
    """Test TimeSeriesStatsPlot with dask-backed xarray."""
    dates, obs, mod = sample_ts_data
    ds = xr.Dataset(
        {
            "obs": (["time"], obs),
            "mod": (["time"], mod),
        },
        coords={"time": dates},
    ).chunk({"time": 5})

    plot = TimeSeriesStatsPlot(ds, col1="obs", col2="mod")
    assert plot.x == "time"

    # Test RMSE with different frequency
    ax = plot.plot(stat="rmse", freq="10h")
    assert ax is not None

    # Verify provenance
    assert "history" in plot.data.attrs
    assert "Generated TimeSeriesStatsPlot" in plot.data.attrs["history"]

    # Check values (RMSE should be 0.1)
    # We can't easily check the plot values without more effort,
    # but the history and lack of error is good.
    plt.close(plot.fig)


def test_timeseries_stats_multiple_models(sample_ts_data):
    """Test TimeSeriesStatsPlot with multiple model columns."""
    dates, obs, mod = sample_ts_data
    ds = xr.Dataset(
        {
            "obs": (["time"], obs),
            "mod1": (["time"], mod),
            "mod2": (["time"], mod + 0.1),
        },
        coords={"time": dates},
    )

    plot = TimeSeriesStatsPlot(ds, col1="obs", col2=["mod1", "mod2"])
    ax = plot.plot(stat="mae", freq="10h")

    assert len(ax.get_lines()) == 2
    plt.close(plot.fig)


def test_timeseries_stats_invalid_stat(sample_ts_data):
    """Test TimeSeriesStatsPlot with unsupported statistic."""
    dates, obs, mod = sample_ts_data
    df = pd.DataFrame({"time": dates, "obs": obs, "mod": mod})

    plot = TimeSeriesStatsPlot(df, col1="obs", col2="mod")
    with pytest.raises(ValueError, match="is not supported"):
        plot.plot(stat="nonexistent_stat")
    plt.close(plot.fig)


def test_timeseries_labels_spread_and_time_basis():
    import pandas as pd

    from monet_plots.plots.timeseries import TimeSeriesPlot

    frame = pd.DataFrame(
        {
            "time": pd.to_datetime(["2020-01-01", "2020-01-01"], utc=True),
            "obs": [1.0, 3.0],
        }
    )
    plot = TimeSeriesPlot(frame)
    try:
        ax = plot.plot()
        assert ax.get_ylabel() == "obs"
        assert "UTC" in ax.get_xlabel()
        assert "standard deviation" in ax.collections[-1].get_label()
    finally:
        plot.close()


@pytest.mark.parametrize("kind", ["statistics", "errorbar"])
def test_xarray_timeseries_batches_multi_model_summaries(kind):
    from dask.callbacks import Callback

    from monet_plots.plots.timeseries import TimeSeriesErrorBarPlot

    times = pd.date_range("2025-01-01", periods=12, freq="h")
    obs = np.arange(36, dtype=float).reshape(12, 3)
    data = xr.Dataset(
        {
            "obs": (("time", "station"), obs),
            "a": (("time", "station"), obs + 1),
            "b": (("time", "station"), obs - 2),
        },
        coords={"time": times},
    )
    lazy = data.chunk({"time": 3, "station": 2})
    if kind == "statistics":
        plot_type, constructor, options = (
            TimeSeriesStatsPlot,
            {"col1": "obs", "col2": ["a", "b"]},
            {"freq": "3h"},
        )
    else:
        plot_type, constructor, options = TimeSeriesErrorBarPlot, {"y": ["a", "b"]}, {}
    eager_plot = plot_type(data, **constructor)
    lazy_plot = plot_type(lazy, **constructor)
    try:
        starts = []
        with Callback(start=lambda graph: starts.append(len(graph))):
            actual = lazy_plot.plot(**options)
        assert len(starts) == 1
        expected = eager_plot.plot(**options)
        for left, right in zip(actual.lines, expected.lines, strict=True):
            np.testing.assert_allclose(left.get_ydata(), right.get_ydata())
        for left, right in zip(actual.collections, expected.collections, strict=True):
            for a, b in zip(left.get_segments(), right.get_segments(), strict=True):
                np.testing.assert_allclose(a, b)
    finally:
        eager_plot.close()
        lazy_plot.close()


@pytest.mark.parametrize("stat", ["bias", "mae", "rmse"])
def test_time_statistics_weight_each_valid_pair_when_bins_have_missing_data(stat):
    obs = np.array([[1.0, 2.0], [3.0, np.nan], [5.0, 6.0], [np.nan, np.nan]])
    model = np.array([[2.0, 4.0], [6.0, 12.0], [1.0, 8.0], [20.0, 30.0]])
    data = xr.Dataset(
        {
            "obs": (("time", "site"), obs),
            "model": (("time", "site"), model),
        },
        coords={"time": pd.date_range("2025-01-01", periods=4, freq="h")},
    )
    plot = TimeSeriesStatsPlot(
        data.chunk({"time": 1, "site": 1}), col1="obs", col2="model"
    )
    try:
        axis = plot.plot(stat=stat, freq="2h")
        differences = (model - obs).reshape(2, 4)
        expected = {
            "bias": np.nanmean(differences, axis=1),
            "mae": np.nanmean(np.abs(differences), axis=1),
            "rmse": np.sqrt(np.nanmean(differences**2, axis=1)),
        }[stat]
        np.testing.assert_allclose(axis.lines[0].get_ydata(), expected)
    finally:
        plot.close()
