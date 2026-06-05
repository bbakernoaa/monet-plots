import matplotlib.pyplot as plt
import matplotlib.axes
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import dask.array as da

from monet_plots.plots.meteogram import Meteogram


@pytest.fixture
def clear_figures():
    """Clear all existing figures before and after a test."""
    plt.close("all")
    yield
    plt.close("all")


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    df = pd.DataFrame(
        {"temp": np.random.rand(10), "pressure": np.random.rand(10)},
        index=pd.to_datetime(np.arange(10), unit="D"),
    )
    df["time"] = df.index
    return df


@pytest.fixture
def sample_ds():
    """Create a sample xarray Dataset for testing."""
    times = pd.to_datetime(np.arange(10), unit="D")
    ds = xr.Dataset(
        {
            "temp": (["time"], np.random.rand(10)),
            "pressure": (["time"], np.random.rand(10)),
        },
        coords={"time": times},
    )
    return ds


@pytest.fixture
def sample_ds_lazy():
    """Create a sample lazy xarray Dataset for testing."""
    times = pd.to_datetime(np.arange(10), unit="D")
    ds = xr.Dataset(
        {
            "temp": (["time"], da.from_array(np.random.rand(10), chunks=5)),
            "pressure": (["time"], da.from_array(np.random.rand(10), chunks=5)),
        },
        coords={"time": times},
    )
    return ds


def test_meteogram_plot_pandas(clear_figures, sample_df):
    """Test that Meteogram creates a plot with Pandas."""
    plot = Meteogram(data=sample_df, variables=["temp", "pressure"])
    axs = plot.plot()
    assert len(axs) == 2
    assert plot.ax == axs
    assert isinstance(axs[0], matplotlib.axes.Axes)


def test_meteogram_plot_xarray_eager(clear_figures, sample_ds):
    """Test that Meteogram creates a plot with eager Xarray."""
    plot = Meteogram(data=sample_ds, variables=["temp", "pressure"])
    axs = plot.plot()
    assert len(axs) == 2
    assert "history" in plot.df.attrs
    assert "Initialized Meteogram" in plot.df.attrs["history"]
    assert "Generated Meteogram Plot" in plot.df.attrs["history"]


def test_meteogram_plot_xarray_lazy(clear_figures, sample_ds_lazy):
    """Test that Meteogram creates a plot with lazy Xarray."""
    plot = Meteogram(data=sample_ds_lazy, variables=["temp", "pressure"])
    axs = plot.plot()
    assert len(axs) == 2
    assert "history" in plot.df.attrs
    assert "Initialized Meteogram" in plot.df.attrs["history"]


def test_meteogram_backward_compatibility(clear_figures, sample_df):
    """Test backward compatibility with 'df' argument."""
    plot = Meteogram(df=sample_df, variables=["temp"])
    axs = plot.plot()
    assert len(axs) == 1


def test_meteogram_no_data_error():
    """Test that providing no data raises an error."""
    with pytest.raises(ValueError, match="Must provide data to Meteogram."):
        Meteogram(variables=["temp"])


def test_meteogram_dataarray(clear_figures, sample_ds):
    """Test that Meteogram handles DataArray input."""
    da_input = sample_ds["temp"]
    plot = Meteogram(data=da_input, variables=["temp"])
    axs = plot.plot()
    assert len(axs) == 1
    assert isinstance(plot.df, xr.Dataset)
