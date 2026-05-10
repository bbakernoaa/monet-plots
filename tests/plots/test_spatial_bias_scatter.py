import datetime
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr

try:
    import dask.array as da
except ImportError:
    da = None

from monet_plots.plots.spatial_bias_scatter import SpatialBiasScatterPlot


def test_spatial_bias_scatter_plot():
    """Test the SpatialBiasScatterPlot plot method."""
    # Create a sample dataframe
    df = pd.DataFrame(
        {
            "latitude": np.arange(30, 40),
            "longitude": np.arange(-100, -90),
            "CMAQ": np.random.rand(10),
            "Obs": np.random.rand(10),
            "datetime": [datetime.datetime(2020, 1, 1)] * 10,
        }
    )

    # Create a SpatialBiasScatterPlot instance
    plot = SpatialBiasScatterPlot(df, col1="Obs", col2="CMAQ")

    # Call the plot method
    cbar = plot.plot()

    # Assert that the plot objects are created
    assert cbar is not None
    plot.close()


def test_spatial_bias_scatter_on_existing_ax():
    """Test that SpatialBiasScatterPlot can draw on a pre-existing GeoAxes."""
    df = pd.DataFrame(
        {
            "latitude": np.arange(30, 40),
            "longitude": np.arange(-100, -90),
            "CMAQ": np.random.rand(10),
            "Obs": np.random.rand(10),
        }
    )

    # 1. Create a figure and a cartopy GeoAxes
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})

    # 2. Instantiate the plot on the existing axes
    plot = SpatialBiasScatterPlot(df, col1="Obs", col2="CMAQ", ax=ax)

    # 3. Assert that the plot object is using the correct axes
    assert plot.ax is ax

    # 4. Call the plot method
    plot.plot()

    # 5. Assert that a scatter plot was actually created
    assert len(ax.collections) > 0
    plt.close(fig)


@pytest.mark.skipif(da is None, reason="dask not installed")
def test_spatial_bias_scatter_laziness():
    """Verify SpatialBiasScatterPlot handles lazy data without eager compute during init."""
    lon = np.linspace(-120, -70, 100)
    lat = np.linspace(25, 50, 100)
    obs = np.random.rand(100)
    mod = np.random.rand(100)

    ds = xr.Dataset(
        {
            "obs": (["sample"], obs),
            "mod": (["sample"], mod),
            "lat": (["sample"], lat),
            "lon": (["sample"], lon),
        }
    ).chunk({"sample": 50})

    # Initialize plot
    plot = SpatialBiasScatterPlot(ds, "obs", "mod")

    # Verify data is still lazy
    assert hasattr(plot.data.obs.data, "chunks")
    # Verify history was updated
    assert (
        "Initialized monet-plots.SpatialBiasScatterPlot" in plot.data.attrs["history"]
    )
    plot.close()


@pytest.mark.skipif(da is None, reason="dask not installed")
def test_spatial_bias_scatter_plot_parity():
    """Verify SpatialBiasScatterPlot produces an axes when plotted with both eager and lazy data."""
    lon = np.linspace(-120, -70, 10)
    lat = np.linspace(25, 50, 10)
    obs = np.random.rand(10)
    mod = np.random.rand(10)

    ds = xr.Dataset(
        {
            "obs": (["sample"], obs),
            "mod": (["sample"], mod),
            "lat": (["sample"], lat),
            "lon": (["sample"], lon),
        }
    )

    lazy_ds = ds.chunk({"sample": 5})

    # Eager plot
    plot_eager = SpatialBiasScatterPlot(ds, "obs", "mod")
    ax_eager = plot_eager.plot()
    assert ax_eager is not None
    plot_eager.close()

    # Lazy plot
    plot_lazy = SpatialBiasScatterPlot(lazy_ds, "obs", "mod")
    ax_lazy = plot_lazy.plot()
    assert ax_lazy is not None
    plot_lazy.close()
