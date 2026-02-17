# tests/plots/test_diurnal_error.py
import pandas as pd
import numpy as np
import xarray as xr
import dask.array as da
import matplotlib.pyplot as plt
import pytest
from monet_plots.plots.diurnal_error import DiurnalErrorPlot


def test_diurnal_error_plot_pandas():
    """Test DiurnalErrorPlot with Pandas DataFrame (Backward compatibility)."""
    dates = pd.date_range("2023-01-01", periods=100, freq="h")
    df = pd.DataFrame(
        {
            "time": dates,
            "obs": np.random.rand(100) * 10,
            "mod": np.random.rand(100) * 10,
        }
    )

    # Default (month)
    plot = DiurnalErrorPlot(df, obs_col="obs", mod_col="mod")
    ax = plot.plot()
    assert isinstance(ax, plt.Axes)
    assert "second_val" in plot.aggregated.coords
    assert "hour" in plot.aggregated.coords
    plt.close(plot.fig)


def test_diurnal_error_plot_dayofweek():
    """Test second_dim='dayofweek' and metric='error'."""
    dates = pd.date_range("2023-01-01", periods=500, freq="h")
    df = pd.DataFrame(
        {"time": dates, "obs": np.random.rand(500), "mod": np.random.rand(500)}
    )

    plot = DiurnalErrorPlot(
        df, obs_col="obs", mod_col="mod", second_dim="dayofweek", metric="error"
    )
    ax = plot.plot()
    assert isinstance(ax, plt.Axes)
    assert ax.get_ylabel() == "Day of Week"
    plt.close(plot.fig)


def test_diurnal_error_plot_xarray():
    """Test DiurnalErrorPlot with native Xarray Dataset."""
    dates = pd.date_range("2023-01-01", periods=240, freq="h")
    ds = xr.Dataset(
        {
            "obs": (["time"], np.random.rand(240)),
            "mod": (["time"], np.random.rand(240)),
        },
        coords={"time": dates},
    )

    plot = DiurnalErrorPlot(ds, obs_col="obs", mod_col="mod", second_dim="month")
    ax = plot.plot()
    assert isinstance(ax, plt.Axes)
    assert isinstance(plot.aggregated, xr.DataArray)
    # Check dimensions
    assert plot.aggregated.shape == (1, 24)  # 1 month, 24 hours
    plt.close(plot.fig)


def test_diurnal_error_plot_lazy():
    """Test DiurnalErrorPlot with Dask-backed Xarray (Speed/Lazy)."""
    dates = pd.date_range("2023-01-01", periods=240, freq="h")
    # Create dask-backed arrays
    obs_da = da.from_array(np.random.rand(240), chunks=120)
    mod_da = da.from_array(np.random.rand(240), chunks=120)

    ds = xr.Dataset(
        {
            "obs": (["time"], obs_da),
            "mod": (["time"], mod_da),
        },
        coords={"time": dates},
    )

    plot = DiurnalErrorPlot(ds, obs_col="obs", mod_col="mod")

    # The aggregated data should still be lazy (Dask-backed)
    assert hasattr(plot.aggregated.data, "dask")

    # Plotting should trigger computation
    ax = plot.plot()
    assert isinstance(ax, plt.Axes)

    # Verify history/provenance
    assert "Calculated diurnal bias" in plot.aggregated.attrs.get("history", "")
    plt.close(plot.fig)


def test_diurnal_error_hvplot():
    """Test hvplot (Track B) availability."""
    pytest.importorskip("hvplot")
    pytest.importorskip("datashader")
    dates = pd.date_range("2023-01-01", periods=48, freq="h")
    df = pd.DataFrame(
        {"time": dates, "obs": np.random.rand(48), "mod": np.random.rand(48)}
    )

    plot = DiurnalErrorPlot(df, obs_col="obs", mod_col="mod")
    hv_obj = plot.hvplot()

    # Just check it returns a HoloViews object
    assert hasattr(hv_obj, "type") or "holoviews" in str(type(hv_obj)).lower()
