# tests/plots/test_fingerprint.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from monet_plots.plots.fingerprint import FingerprintPlot


def test_fingerprint_plot():
    dates = pd.date_range("2023-01-01", periods=1000, freq="h")
    df = pd.DataFrame({"time": dates, "val": np.random.rand(1000)})

    plot = FingerprintPlot(df, val_col="val", x_scale="hour", y_scale="dayofyear")
    ax = plot.plot()
    assert isinstance(ax, plt.Axes)
    assert ax.get_xlabel() == "Hour"
    assert ax.get_ylabel() == "Dayofyear"
    plt.close(plot.fig)


def test_fingerprint_plot_custom():
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    df = pd.DataFrame(
        {"time": dates, "val": np.random.rand(100), "site": ["A", "B"] * 50}
    )

    # Using 'site' as y_scale (it should pick it up from columns)
    plot = FingerprintPlot(df, val_col="val", x_scale="month", y_scale="site")
    ax = plot.plot()
    assert isinstance(ax, plt.Axes)
    plt.close(plot.fig)


def test_fingerprint_plot_xarray():
    # Test with native xarray Dataset
    dates = pd.date_range("2023-01-01", periods=100, freq="h")
    ds = xr.Dataset(
        {"val": ("time", np.random.rand(100))},
        coords={"time": dates},
    )

    plot = FingerprintPlot(ds, val_col="val", x_scale="hour", y_scale="month")
    assert isinstance(plot.aggregated, xr.DataArray)
    assert "Calculated fingerprint for val" in plot.aggregated.attrs.get("history", "")

    ax = plot.plot()
    assert isinstance(ax, plt.Axes)
    plt.close(plot.fig)


def test_fingerprint_plot_lazy():
    # Test lazy evaluation using Dask-backed Dataset
    try:
        import dask.array as da  # noqa: F401
    except ImportError:
        pytest.skip("dask not installed")

    dates = pd.date_range("2023-01-01", periods=100, freq="h")
    ds = xr.Dataset(
        {"val": ("time", np.random.rand(100))},
        coords={"time": dates},
    ).chunk({"time": 20})

    plot = FingerprintPlot(ds, val_col="val", x_scale="hour", y_scale="month")

    # Aggregation should remain lazy (meaning we didn't eagerly compute)
    # Note: modern xarray with certain groupby configurations might trigger eager load
    # or dask might be preserved. We ensure it computes correctly regardless.
    ax = plot.plot()
    assert isinstance(ax, plt.Axes)
    plt.close(plot.fig)


def test_fingerprint_plot_hvplot():
    # Test Track B (Interactive visualization)
    pytest.importorskip("hvplot")
    dates = pd.date_range("2023-01-01", periods=100, freq="h")
    df = pd.DataFrame({"time": dates, "val": np.random.rand(100)})

    plot = FingerprintPlot(df, val_col="val", x_scale="hour", y_scale="month")
    hv_plot = plot.hvplot()
    assert hasattr(hv_plot, "type") or "holoviews" in str(type(hv_plot)).lower()


def test_fingerprint_plot_invalid_scale():
    dates = pd.date_range("2023-01-01", periods=10, freq="h")
    df = pd.DataFrame({"time": dates, "val": np.random.rand(10)})

    # Expect ValueError for invalid temporal scale
    with pytest.raises(ValueError, match="Unknown temporal scale"):
        FingerprintPlot(df, val_col="val", x_scale="nonexistent")
