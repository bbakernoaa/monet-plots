import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from cartopy.mpl.geoaxes import GeoAxes

from monet_plots.plots.facet_grid import SpatialFacetGridPlot
from monet_plots.plots.spatial_imshow import SpatialImshowPlot


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
    axes = grid.grid.axs if hasattr(grid.grid, "axs") else grid.grid.axes
    assert len(axes.flatten()) == 2
    assert isinstance(axes.flatten()[0], GeoAxes)
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
    _axes = grid.grid.axs if hasattr(grid.grid, "axs") else grid.grid.axes
    titles = [ax.get_title() for ax in _axes.flatten()]
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
    _axes = grid.grid.axs if hasattr(grid.grid, "axs") else grid.grid.axes
    for ax in _axes.flatten():
        assert len(ax.images) > 0
    plt.close()


def test_lazy_facet_loads_shared_scale_and_panel_values_together():
    from dask.callbacks import Callback

    data = xr.DataArray(
        np.arange(48).reshape(4, 3, 4),
        dims=("time", "lat", "lon"),
        coords={
            "time": np.arange(4),
            "lat": [30, 33, 36],
            "lon": [-105, -100, -95, -90],
        },
    ).chunk({"time": 1, "lat": 2, "lon": 2})
    grid = SpatialFacetGridPlot(data, col="time", col_wrap=2)
    try:
        runs = []
        with Callback(start=lambda graph: runs.append(len(graph))):
            grid.map_monet(SpatialImshowPlot, x="lon", y="lat", coastlines=False)
        assert len(runs) == 1
        assert grid.grid.data.chunks is not None
        for i, ax in enumerate(grid.grid.axs.flat):
            image = ax.images[0]
            np.testing.assert_allclose(image.get_array(), data.isel(time=i).values)
            assert image.get_clim() == (0, 47)
    finally:
        grid.close()
