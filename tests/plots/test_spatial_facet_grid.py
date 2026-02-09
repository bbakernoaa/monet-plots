import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from monet_plots.plots.facet_grid import SpatialFacetGridPlot
from monet_plots.plots.spatial_imshow import SpatialImshowPlot
from cartopy.mpl.geoaxes import GeoAxes


def test_spatial_facet_grid_init():
    lats = np.linspace(30, 40, 5)
    lons = np.linspace(-100, -90, 5)
    data = xr.DataArray(
        np.random.rand(2, 5, 5),
        coords={
            "time": [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-02")],
            "lat": lats,
            "lon": lons,
        },
        dims=["time", "lat", "lon"],
        name="test",
    )

    grid = SpatialFacetGridPlot(data, col="time")
    assert len(grid.grid.axes.flatten()) == 2
    assert isinstance(grid.grid.axes.flatten()[0], GeoAxes)
    plt.close()


def test_spatial_facet_grid_dataset_variable():
    lats = np.linspace(30, 40, 5)
    lons = np.linspace(-100, -90, 5)
    ds = xr.Dataset(
        {
            "temp": (["lat", "lon"], np.random.rand(5, 5)),
            "pres": (["lat", "lon"], np.random.rand(5, 5)),
        },
        coords={"lat": lats, "lon": lons},
    )
    ds["temp"].attrs["long_name"] = "Temperature"

    grid = SpatialFacetGridPlot(ds, col="variable")
    titles = [ax.get_title() for ax in grid.grid.axes.flatten()]
    assert "Temperature" in titles[0]
    assert "pres" in titles[1]
    plt.close()


def test_spatial_facet_grid_map_monet():
    lats = np.linspace(30, 40, 5)
    lons = np.linspace(-100, -90, 5)
    data = xr.DataArray(
        np.random.rand(2, 5, 5),
        coords={
            "time": [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-02")],
            "lat": lats,
            "lon": lons,
        },
        dims=["time", "lat", "lon"],
        name="test",
    )

    grid = SpatialFacetGridPlot(data, col="time")
    grid.map_monet(SpatialImshowPlot, coastlines=True)

    # Check that each axis has images (from imshow)
    for ax in grid.grid.axes.flatten():
        assert len(ax.images) > 0
    plt.close()
