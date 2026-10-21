"""Compatibility between unified data arguments and legacy spatial entry points."""

import numpy as np
import pytest
import xarray as xr

from monet_plots.plots.spatial_contour import SpatialContourPlot
from monet_plots.plots.spatial_imshow import SpatialImshowPlot
from monet_plots.plots.spatial_mesh import SpatialMeshPlot


@pytest.mark.parametrize(
    "plot_class", [SpatialContourPlot, SpatialImshowPlot, SpatialMeshPlot]
)
@pytest.mark.parametrize("argument", ["data", "modelvar"])
def test_spatial_data_alias_and_metadata(plot_class, argument):
    field = xr.DataArray(
        np.arange(12).reshape(3, 4),
        dims=("lat", "lon"),
        coords={"lat": [30, 35, 40], "lon": [-110, -100, -90, -80]},
        attrs={"long_name": "Surface concentration"},
    )
    plot = plot_class(**{argument: field}, style=None)
    try:
        plot.plot()
        assert plot.data is plot.modelvar
        assert plot.ax.get_title() == "Surface concentration"
    finally:
        plot.close()
