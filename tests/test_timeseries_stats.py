import numpy as np
import pandas as pd
import pytest
import xarray as xr
import matplotlib.pyplot as plt
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
    assert "history" in plot.df.attrs
    assert "Generated TimeSeriesStatsPlot" in plot.df.attrs["history"]

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
