import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from matplotlib import pyplot as plt

from monet_plots.plots.meteogram import Meteogram


@pytest.fixture
def clear_figures():
    """Clear all existing figures before and after a test."""
    plt.close("all")
    yield
    plt.close("all")


@pytest.fixture
def sample_ds():
    """Create a sample lazy xarray Dataset for testing."""
    time = pd.date_range("2023-01-01", periods=24, freq="h")
    temp = da.random.random((24,), chunks=(12,))
    pres = da.random.random((24,), chunks=(12,))
    ds = xr.Dataset(
        {"temp": (("time",), temp), "pres": (("time",), pres)},
        coords={"time": time},
    )
    return ds


@pytest.fixture
def sample_df():
    """Create a sample pandas DataFrame for testing."""
    time = pd.date_range("2023-01-01", periods=24, freq="h")
    return pd.DataFrame(
        {"temp": np.random.rand(24), "pres": np.random.rand(24), "time": time}
    )


def test_meteogram_lazy_xarray(clear_figures, sample_ds):
    """Test Meteogram with lazy xarray Dataset."""
    plot = Meteogram(data=sample_ds, variables=["temp", "pres"], x="time")

    # Verify provenance
    assert "Initialized Meteogram" in plot.data.attrs["history"]

    axes = plot.plot()
    assert len(axes) == 2
    assert "Generated static plot" in plot.data.attrs["history"]

    # Verify data is still lazy
    assert isinstance(plot.data.temp.data, da.Array)


def test_meteogram_pandas(clear_figures, sample_df):
    """Test Meteogram with pandas DataFrame."""
    plot = Meteogram(data=sample_df, variables=["temp", "pres"], x="time")

    axes = plot.plot()
    assert len(axes) == 2


def test_meteogram_hvplot_smoke(sample_ds):
    """Smoke test for hvplot method."""
    pytest.importorskip("hvplot")
    pytest.importorskip("holoviews")

    plot = Meteogram(data=sample_ds, variables=["temp", "pres"], x="time")
    hv_plot = plot.hvplot()

    import holoviews as hv

    assert isinstance(hv_plot, hv.Layout)


def test_meteogram_eager_lazy_consistency(clear_figures, sample_ds):
    """Verify that eager and lazy paths produce consistent results."""
    # Lazy path
    plot_lazy = Meteogram(data=sample_ds, variables=["temp"])
    axes_lazy = plot_lazy.plot()
    line_lazy = axes_lazy[0].get_lines()[0]
    y_lazy = line_lazy.get_ydata()

    # Eager path
    ds_eager = sample_ds.compute()
    plot_eager = Meteogram(data=ds_eager, variables=["temp"])
    axes_eager = plot_eager.plot()
    line_eager = axes_eager[0].get_lines()[0]
    y_eager = line_eager.get_ydata()

    np.testing.assert_allclose(y_lazy, y_eager)
