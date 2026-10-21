"""Scientific correctness checks: placement, shared scales, metadata and export."""

import cartopy.crs as ccrs
import dask.array as da
import matplotlib.pyplot as plt
import monet_plots as mp
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from dask.callbacks import Callback
from matplotlib.colors import BoundaryNorm, TwoSlopeNorm
from monet_plots.geo import identify_coords
from monet_plots.plots.spatial_overlay import SpatialOverlayPlot


@pytest.fixture(autouse=True)
def clean_figures():
    yield
    plt.close("all")


@pytest.fixture
def field():
    return xr.DataArray(
        np.arange(12.0).reshape(3, 4),
        dims=("lat", "lon"),
        coords={"lat": [30, 35, 40], "lon": [-110, -100, -90, -80]},
        name="o3",
        attrs={"units": "ppb", "long_name": "Ozone"},
    )


def facets(field):
    return xr.concat(
        [field, field + 100, field + 200], dim=pd.Index(["a", "b", "c"], name="model")
    )


@pytest.mark.parametrize("kind", ["pcolormesh", "contourf", "imshow", "scatter"])
def test_facets_share_real_norm_and_one_colorbar(field, kind):
    data = facets(field)
    if kind == "scatter":
        data = data.stack(sample=("lat", "lon"))
    result = mp.spatial(
        data,
        kind=kind,
        col="model",
        col_wrap=2,
        coastlines=False,
        cmap="plasma",
        vmin=0,
        vmax=250,
        size=2,
        aspect=1.5,
    )
    assert len(result.mappables) == 3
    norm = result.mappables[0].norm
    assert all(artist.norm is norm for artist in result.mappables)
    assert norm.vmin == 0 and norm.vmax == 250
    assert all(artist.cmap.name == "plasma" for artist in result.mappables)
    assert len(result.colorbars) == 1
    assert result.colorbars[0].mappable is result.mappables[-1]
    assert len(result.fig.axes) == 5  # 4 panels, 1 colorbar
    assert not result.axes.flat[-1].get_visible()
    assert result.fig.get_size_inches()[1] == 4
    assert result.colorbars[0].ax.get_ylabel() == "Ozone (ppb)"


def test_global_limits_not_per_panel(field):
    result = mp.spatial(facets(field), col="model", coastlines=False, colorbar=False)
    assert result.mappables[0].norm.vmin == 0
    assert result.mappables[0].norm.vmax == 211
    assert len(result.colorbars) == 0


@pytest.mark.parametrize("kind", ["contourf", "pcolormesh"])
def test_custom_norm_and_irregular_levels(field, kind):
    levels = [-1, 1, 10, 50, 250]
    norm = BoundaryNorm(levels, 256)
    options = {"levels": levels} if kind == "contourf" else {}
    result = mp.spatial(
        facets(field), col="model", kind=kind, norm=norm, coastlines=False, **options
    )
    assert all(artist.norm is norm for artist in result.mappables)
    if kind == "contourf":
        np.testing.assert_equal(result.colorbars[0].boundaries, levels)


def test_raster_pixel_bounds_and_viewport_independent(field):
    result = mp.spatial(
        field, kind="imshow", extent=[-120, -70, 20, 50], coastlines=False
    )
    np.testing.assert_allclose(result.mappable.get_extent(), [-115, -75, 27.5, 42.5])
    np.testing.assert_allclose(
        result.ax.get_extent(ccrs.PlateCarree()), [-120, -70, 20, 50]
    )
    np.testing.assert_equal(result.mappable.get_array(), field.values)


def test_raster_transposes_and_sorts_without_modifying_input(field):
    data = field.isel(lat=slice(None, None, -1)).transpose("lon", "lat")
    before = data.copy(deep=True)
    result = mp.spatial(data, kind="imshow", coastlines=False)
    np.testing.assert_equal(result.mappable.get_array(), field.values)
    xr.testing.assert_identical(data, before)


def test_curvilinear_mesh_and_imshow_rejection(field):
    lon, lat = np.meshgrid(field.lon, field.lat)
    data = xr.DataArray(
        field.values,
        dims=("j", "i"),
        coords={
            "longitude": (("j", "i"), lon + 0.1 * lat),
            "latitude": (("j", "i"), lat + 0.01 * lon),
        },
    )
    result = mp.spatial(data, coastlines=False)
    np.testing.assert_allclose(
        result.mappable.get_coordinates()[1, 1],
        [0.5 * (lon[0, 0] + lon[0, 1]) + 0.1 * 32.5, 32.5 + 0.01 * -105],
    )
    with pytest.raises(ValueError, match="regular 1D"):
        mp.spatial(data, kind="imshow", coastlines=False)


def test_projected_grid_transform_and_geographic_extent(field):
    crs = ccrs.LambertConformal(central_longitude=-100, central_latitude=35)
    data = field.rename(lat="northing", lon="easting").assign_coords(
        northing=[-100000, 0, 100000], easting=[-150000, -50000, 50000, 150000]
    )
    data.northing.attrs["standard_name"] = "projection_y_coordinate"
    data.easting.attrs["standard_name"] = "projection_x_coordinate"
    result = mp.spatial(
        data,
        data_crs=crs,
        projection=ccrs.Mercator(),
        extent=[-110, -90, 25, 45],
        coastlines=False,
    )
    expected = crs._as_mpl_transform(result.ax)
    points = np.array([[0, 0], [100000, 100000]])
    np.testing.assert_allclose(
        result.mappable.get_transform().transform(points), expected.transform(points)
    )
    assert result.ax.get_extent(ccrs.PlateCarree())[0] == pytest.approx(-110, abs=0.1)


def test_coordinate_detection_does_not_match_substrings(field):
    ds = field.to_dataset().assign(
        index=("lon", [1, 2, 3, 4]), humidity=("lat", [2, 3, 4])
    )
    assert identify_coords(ds) == ("lon", "lat")
    ds = ds.assign_coords(longitude=("lon", field.lon.values))
    with pytest.raises(ValueError, match="unique longitude"):
        identify_coords(ds)
    assert identify_coords(ds, "longitude", "lat") == ("longitude", "lat")


def test_station_pairs_not_independently_sorted():
    data = xr.DataArray(
        [10, 20, 30],
        dims="site",
        coords={"lon": ("site", [-80, -100, -90]), "lat": ("site", [30, 40, 35])},
    )
    result = mp.spatial(data, kind="scatter", coastlines=False)
    np.testing.assert_equal(
        result.mappable.get_offsets(), [[-80, 30], [-100, 40], [-90, 35]]
    )
    np.testing.assert_equal(result.mappable.get_array(), [10, 20, 30])


def test_dataset_units_require_independent_scales(field):
    ds = xr.Dataset({"ozone": field, "temp": (field + 273).assign_attrs(units="K")})
    with pytest.raises(ValueError, match="incompatible units"):
        mp.spatial(ds, col="variable", coastlines=False)
    result = mp.spatial(ds, col="variable", colorbar="each", coastlines=False)
    assert len(result.colorbars) == 2
    assert result.colorbars[0].ax.get_ylabel() == "Ozone (ppb)"
    assert result.colorbars[1].ax.get_ylabel() == "Ozone (K)"
    assert result.mappables[0].norm.vmin != result.mappables[1].norm.vmin


def test_lazy_init_no_computation_and_inputs_preserved(field):
    data = facets(field).chunk({"model": 1, "lat": 2})
    before = data.copy(deep=True)
    tasks = []
    with Callback(posttask=lambda *args: tasks.append(args)):
        result = mp.SpatialFacetGridPlot(data, col="model", coastlines=False)
    assert tasks == []
    result.plot()
    assert isinstance(result.data.data, da.Array)
    xr.testing.assert_identical(data, before)


def test_two_by_four_comparison_export_and_observations(field, tmp_path):
    data = xr.concat(
        [field + i * 5 for i in range(4)],
        dim=pd.Index(["DJF", "MAM", "JJA", "SON"], name="season"),
    )
    data = xr.concat([data, data + 10], dim=pd.Index(["A", "B"], name="model"))
    obs = pd.DataFrame({"lon": [-100, -90], "lat": [35, 40], "o3": [20, 30]})
    result = mp.spatial(
        data,
        row="model",
        col="season",
        projection=ccrs.LambertConformal(),
        observations=obs,
        obs_var="o3",
        coastlines=False,
        panel_labels=True,
        vmin=0,
        vmax=50,
        size=1.8,
        rasterized=True,
    )
    assert result.axes.shape == (2, 4)
    assert len(result.overlays) == 8
    assert all(overlay.norm is result.mappables[0].norm for overlay in result.overlays)
    assert all(len(ax.texts) == 1 for ax in result.axes.flat)
    for suffix in ["png", "pdf"]:
        path = tmp_path / f"comparison.{suffix}"
        result.save(path, dpi=80)
        assert path.stat().st_size > 1000


def test_observation_overlay_preserves_zero_limit(field):
    obs = pd.DataFrame({"lon": [-100], "lat": [35], "obs": [-5]})
    result = SpatialOverlayPlot(field, obs, obs_var="obs", coastlines=False)
    result.plot(kind="imshow", vmin=0, vmax=20)
    assert result.mappable.norm.vmin == 0
    assert result.observation_artist.norm is result.mappable.norm
    assert len(result.colorbars) == 1


def test_class_redirect_renders_originating_plotter(field):
    result = mp.SpatialContourPlot(facets(field), col="model", coastlines=False)
    result.plot(levels=[0, 10, 50, 250])
    assert all(isinstance(p, mp.SpatialContourPlot) for p in result.plotters)
    assert len(result.mappables) == 3


def test_bias_facets_zero_and_negative():
    ds = xr.Dataset(
        {
            "obs": (("model", "site"), [[0, 0], [0, 0]]),
            "mod": (("model", "site"), [[0, 0], [-1, -2]]),
        },
        coords={
            "model": ["A", "B"],
            "site": [0, 1],
            "lon": ("site", [-100, -90]),
            "lat": ("site", [35, 40]),
        },
    )
    result = mp.SpatialBiasScatterPlot(ds, "obs", "mod", col="model", coastlines=False)
    result.plot()
    assert result.mappables[0].norm.vmin == -2
    assert result.mappables[0].norm.vmax == 2
    assert result.mappables[0].norm is result.mappables[1].norm


def test_pandas_facets_keep_duplicate_stations():
    df = pd.DataFrame(
        {
            "model": ["A", "A", "B", "B"],
            "lon": [-100] * 4,
            "lat": [35] * 4,
            "o3": [1, 2, 10, 20],
        }
    )
    result = mp.spatial(
        df, kind="scatter", col="model", var_name="o3", coastlines=False
    )
    assert [len(artist.get_offsets()) for artist in result.mappables] == [2, 2]
    assert result.mappables[0].norm is result.mappables[1].norm


def test_missing_panel_and_dateline(field):
    data = facets(field)
    data.loc[{"model": "b"}] = np.nan
    result = mp.spatial(data, col="model", coastlines=False)
    assert len(result.mappables) == 3
    dateline = field.assign_coords(lon=[170, 180, -170, -160])
    result = mp.spatial(
        dateline,
        kind="imshow",
        coastlines=False,
        projection=ccrs.PlateCarree(central_longitude=180),
    )
    assert abs(
        result.mappable.get_extent()[1] - result.mappable.get_extent()[0]
    ) == pytest.approx(40)


def test_diverging_scale(field):
    norm = TwoSlopeNorm(vcenter=0, vmin=-20, vmax=250)
    result = mp.spatial(facets(field), col="model", coastlines=False, norm=norm)
    assert result.mappables[0].norm is norm


def test_projected_metadata_requires_explicit_crs(field):
    data = field.rename(lat="y", lon="x")
    data.x.attrs["standard_name"] = "projection_x_coordinate"
    with pytest.raises(ValueError, match="explicit data_crs"):
        mp.spatial(data, coastlines=False)
    with pytest.raises(ValueError, match="explicit data_crs"):
        mp.spatial(facets(data), col="model", coastlines=False)


def test_log_scale_ignores_nonpositive_values(field):
    from matplotlib.colors import LogNorm

    result = mp.spatial(facets(field), col="model", coastlines=False, norm=LogNorm())
    assert result.mappables[0].norm.vmin == 1
    assert result.mappables[0].norm.vmax == 211


def test_shared_contour_boundary_norm_supplies_levels(field):
    levels = [-1, 1, 10, 50, 250]
    result = mp.spatial(
        facets(field),
        col="model",
        kind="contourf",
        coastlines=False,
        norm=BoundaryNorm(levels, 256),
    )
    np.testing.assert_equal(result.mappables[0].levels, levels)


def test_observation_units_and_automatic_facet_selection(field):
    obs = xr.DataArray(
        [[10, 20], [20, 30], [30, 40]],
        dims=("model", "site"),
        coords={
            "model": ["a", "b", "c"],
            "site": [0, 1],
            "lon": ("site", [-100, -90]),
            "lat": ("site", [35, 40]),
        },
        attrs={"units": "ppb"},
    )
    result = mp.spatial(facets(field), col="model", observations=obs, coastlines=False)
    for index, overlay in enumerate(result.overlays):
        np.testing.assert_equal(overlay.get_array(), obs.isel(model=index).values)
    with pytest.raises(ValueError, match="units differ"):
        mp.spatial(field, observations=obs.isel(model=0).assign_attrs(units="K"))


def test_incompatible_variable_dimensions_are_not_broadcast(field):
    data = xr.Dataset({"a": field, "b": ("site", [1, 2])})
    with pytest.raises(ValueError, match="matching dimensions"):
        mp.spatial(data, col="variable", coastlines=False)


def test_bias_scatter_honors_matplotlib_aliases():
    data = pd.DataFrame(
        {"lon": [-100, -90], "lat": [35, 40], "obs": [0, 0], "mod": [-1, 1]}
    )
    result = mp.SpatialBiasScatterPlot(data, "obs", "mod", coastlines=False)
    result.plot(edgecolor="red", linewidth=2)
    assert result.mappable.get_linewidths()[0] == 2


def test_single_map_tabular_selection_and_existing_axes(field):
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    result = mp.spatial(
        field.to_dataset(), var_name="o3", ax=ax, coastlines=False, colorbar=False
    )
    assert result.fig is fig and result.axes[0, 0] is ax
    points = field.to_dataframe().reset_index()
    result = mp.spatial(points, kind="scatter", var_name="o3", coastlines=False)
    assert len(result.mappable.get_array()) == 12


def test_legacy_plotargs_apply_to_shared_facet_scale(field):
    result = mp.SpatialImshowPlot(
        facets(field),
        col="model",
        coastlines=False,
        plotargs={"vmin": 0, "vmax": 300, "cmap": "plasma"},
    )
    result.plot()
    assert all(artist.norm.vmax == 300 for artist in result.mappables)
    assert all(artist.cmap.name == "plasma" for artist in result.mappables)


def test_gridline_labels_are_outer_only_and_can_be_disabled(field):
    result = mp.spatial(
        facets(field),
        col="model",
        col_wrap=2,
        coastlines=False,
        gridlines={"draw_labels": True},
    )
    assert not result.plotters[0].gridliner.bottom_labels
    assert result.plotters[1].gridliner.bottom_labels
    assert not result.plotters[1].gridliner.left_labels
    assert result.plotters[2].gridliner.bottom_labels
    assert all(not p.gridliner.x_inline for p in result.plotters)
    result = mp.spatial(
        facets(field), col="model", coastlines=False, gridlines={"draw_labels": False}
    )
    assert not any(p.gridliner.left_labels for p in result.plotters)


def test_imshow_accepts_large_regular_float32_coordinates():
    from monet_plots.geo import raster_extent

    coords = np.linspace(-125, -65, 1024, dtype=np.float32)
    field = xr.DataArray(
        np.zeros((2, 1024)),
        dims=("lat", "lon"),
        coords={"lat": np.array([30, 35], dtype=np.float32), "lon": coords},
    )
    extent = raster_extent(field, "lon", "lat")
    assert extent[0] == pytest.approx(-125 - 30 / 1023)
    assert extent[1] == pytest.approx(-65 + 30 / 1023)
    result = mp.spatial(field, kind="imshow", coastlines=False, colorbar=False)
    assert result.mappable.get_array().shape == (2, 1024)
    irregular = field.assign_coords(lon=coords + np.where(np.arange(1024) % 2, 0.01, 0))
    with pytest.raises(ValueError, match="regularly spaced"):
        raster_extent(irregular, "lon", "lat")


def test_mesh_computes_each_dask_panel_once(field):
    from dask.callbacks import Callback

    data = facets(field).chunk({"model": 1, "lat": 1, "lon": 2})
    evaluations = []
    with Callback(start=lambda graph: evaluations.append(graph)):
        result = mp.spatial(data, col="model", vmin=0, vmax=250, coastlines=False)
    assert len(evaluations) == 3
    # Repeated canvas draws reuse concrete artists rather than rerunning data graphs.
    with Callback(
        start=lambda graph: pytest.fail("Unexpected recomputation during draw")
    ):
        result.fig.canvas.draw()
        result.fig.canvas.draw()
