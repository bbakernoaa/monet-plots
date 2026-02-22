import numpy as np
import pandas as pd
import pytest
import xarray as xr
import dask.array as da
from monet_plots.plots import DiurnalErrorPlot


def test_diurnal_error_vectorized_numpy():
    """Verify DiurnalErrorPlot works with NumPy-backed Xarray."""
    dates = pd.date_range("2023-01-01", periods=100, freq="h")
    obs = np.random.rand(100)
    mod = np.random.rand(100)

    ds = xr.Dataset(
        {
            "obs": (["time"], obs),
            "mod": (["time"], mod),
        },
        coords={"time": dates},
    )

    plot = DiurnalErrorPlot(ds, obs_col="obs", mod_col="mod")

    # Check aggregation results
    assert plot.aggregated is not None
    assert plot.aggregated.dims == ("second_val", "hour")
    assert plot.aggregated.shape == (1, 24)  # Only Jan 2023
    assert "Calculated diurnal bias" in plot.aggregated.attrs["history"]
    assert "(monet-plots)" in plot.aggregated.attrs["history"]


def test_diurnal_error_vectorized_dask():
    """Verify DiurnalErrorPlot works lazily with Dask-backed Xarray."""
    dates = pd.date_range("2023-01-01", periods=1000, freq="h")
    obs = da.random.random(1000, chunks=200)
    mod = da.random.random(1000, chunks=200)

    ds = xr.Dataset(
        {
            "obs": (["time"], obs),
            "mod": (["time"], mod),
        },
        coords={"time": dates},
    )

    plot = DiurnalErrorPlot(ds, obs_col="obs", mod_col="mod", second_dim="month")

    # Verify laziness
    assert hasattr(plot.aggregated.data, "chunks")

    # Verify shape before compute
    assert plot.aggregated.dims == ("second_val", "hour")
    # 1000 hours spans Jan and Feb
    assert plot.aggregated.shape == (2, 24)

    # Compute and check values
    res = plot.aggregated.compute()
    assert not np.isnan(res).any()
    assert "Calculated diurnal bias" in res.attrs["history"]


def test_diurnal_error_metrics():
    """Verify both 'bias' and 'error' metrics work."""
    dates = pd.date_range("2023-01-01", periods=24, freq="h")
    # mod = obs + 1 -> bias should be 1, error should be 1
    obs = np.zeros(24)
    mod = np.ones(24)

    ds = xr.Dataset(
        {
            "obs": (["time"], obs),
            "mod": (["time"], mod),
        },
        coords={"time": dates},
    )

    plot_bias = DiurnalErrorPlot(ds, obs_col="obs", mod_col="mod", metric="bias")
    assert np.allclose(plot_bias.aggregated.values, 1.0)

    plot_err = DiurnalErrorPlot(ds, obs_col="obs", mod_col="mod", metric="error")
    assert np.allclose(plot_err.aggregated.values, 1.0)


def test_diurnal_error_second_dim_custom():
    """Verify custom second dimension works."""
    dates = pd.date_range("2023-01-01", periods=48, freq="h")
    ds = xr.Dataset(
        {
            "obs": (["time"], np.random.rand(48)),
            "mod": (["time"], np.random.rand(48)),
            "region": (["time"], np.repeat(["North", "South"], 24)),
        },
        coords={"time": dates},
    )

    plot = DiurnalErrorPlot(ds, obs_col="obs", mod_col="mod", second_dim="region")
    assert plot.aggregated.shape == (2, 24)
    assert set(plot.aggregated.second_val.values) == {"North", "South"}
    assert plot.second_label == "region"


def test_diurnal_error_pandas_fallback():
    """Verify backward compatibility with Pandas."""
    dates = pd.date_range("2023-01-01", periods=100, freq="h")
    df = pd.DataFrame(
        {"time": dates, "obs": np.random.rand(100), "mod": np.random.rand(100)}
    )

    plot = DiurnalErrorPlot(df, obs_col="obs", mod_col="mod")
    assert isinstance(plot.aggregated, xr.DataArray)
    assert plot.aggregated.shape == (1, 24)


if __name__ == "__main__":
    pytest.main([__file__])
