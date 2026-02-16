import numpy as np
import pandas as pd
import xarray as xr
import pytest
from monet_plots.plots.spatial_imshow import SpatialImshowPlot
from monet_plots.plots.spatial_contour import SpatialContourPlot
from monet_plots.plots.spatial_bias_scatter import SpatialBiasScatterPlot
from monet_plots.plots.spatial import SpatialTrack
from monet_plots.plots.scatter import ScatterPlot


@pytest.fixture
def sample_da():
    lon = np.linspace(-120, -70, 10)
    lat = np.linspace(25, 50, 10)
    data = np.random.rand(10, 10)
    da = xr.DataArray(data, coords=[lat, lon], dims=["lat", "lon"], name="test_var")
    da.lat.attrs["units"] = "degrees_north"
    da.lon.attrs["units"] = "degrees_east"
    return da


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "latitude": [30, 35, 40],
            "longitude": [-100, -90, -80],
            "obs": [1, 2, 3],
            "fcst": [1.1, 2.1, 3.1],
        }
    )


def test_spatial_imshow_hvplot(sample_da):
    # Pass ax=None to avoid creating a new figure if possible,
    # but BasePlot creates one by default.
    plot_obj = SpatialImshowPlot(sample_da)
    hv_plot = plot_obj.hvplot()
    # hvplot returns a HoloViews object (Element, DynamicMap, etc.)
    assert hasattr(hv_plot, "opts")
    assert "Generated interactive SpatialImshowPlot" in sample_da.attrs.get(
        "history", ""
    )


def test_spatial_contour_hvplot(sample_da):
    plot_obj = SpatialContourPlot(sample_da)
    hv_plot = plot_obj.hvplot()
    assert hasattr(hv_plot, "opts")
    assert "Generated interactive SpatialContourPlot" in sample_da.attrs.get(
        "history", ""
    )


def test_spatial_bias_scatter_hvplot(sample_df):
    plot_obj = SpatialBiasScatterPlot(sample_df, col1="obs", col2="fcst")
    hv_plot = plot_obj.hvplot()
    assert hasattr(hv_plot, "opts")


def test_spatial_track_hvplot():
    # Create trajectory data
    track_da = xr.DataArray(
        np.random.rand(5),
        coords={
            "lon": (["sample"], [-100, -95, -90, -85, -80]),
            "lat": (["sample"], [30, 32, 34, 36, 38]),
        },
        dims=["sample"],
        name="track_var",
    )
    plot_obj = SpatialTrack(track_da)
    hv_plot = plot_obj.hvplot()
    assert hasattr(hv_plot, "opts")


def test_scatter_hvplot(sample_da):
    # Reshape for scatter
    df = sample_da.to_dataframe().reset_index()
    plot_obj = ScatterPlot(df, x="lon", y="test_var")
    hv_plot = plot_obj.hvplot()
    assert hasattr(hv_plot, "opts")
