"""Correctness and lazy-computation contracts for observation/model/bias maps."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from dask.callbacks import Callback
from matplotlib.colors import Normalize, TwoSlopeNorm

import monet_plots as mp


@pytest.fixture(autouse=True)
def clean_figures():
    yield
    plt.close("all")


@pytest.fixture
def fields():
    obs = xr.DataArray(
        np.arange(12, dtype=float).reshape(3, 4),
        dims=("lat", "lon"),
        coords={"lat": [30, 35, 40], "lon": [-110, -100, -90, -80]},
        name="ozone",
        attrs={"units": "ppb", "long_name": "Ozone"},
    )
    return obs, obs + 2


@pytest.mark.parametrize("kind", ["pcolormesh", "contourf", "imshow"])
def test_two_scales_and_actual_colorbar_mappables(fields, kind):
    obs, model = fields
    result = mp.compare_spatial(obs, model, kind=kind, coastlines=False)
    assert result.axes.shape == (1, 3)
    assert len(result.colorbars) == 2
    assert result.mappables[0].norm is result.mappables[1].norm
    assert result.field_norm.vmin == 0 and result.field_norm.vmax == 13
    assert result.mappables[2].norm is result.bias_norm
    assert (
        result.bias_norm(-2) == 0
        and result.bias_norm(0) == 0.5
        and result.bias_norm(2) == 1
    )
    assert result.colorbars[0].mappable is result.mappables[0]
    assert result.colorbars[1].mappable is result.mappables[2]
    assert result.colorbars[0].ax.get_xlabel() == "Ozone (ppb)"
    np.testing.assert_array_equal(result.bias, np.full((3, 4), 2))
    assert int(result.paired_counts) == 12


def test_static_observations_repeated_over_model_dimension(fields):
    obs, model = fields
    models = xr.concat([model, model - 5], dim=xr.IndexVariable("model", ["A", "B"]))
    result = mp.compare_spatial(obs, models, row="model", coastlines=False)
    assert result.axes.shape == (2, 3)
    assert result.bias_norm.vmin == -3 and result.bias_norm.vmax == 3
    np.testing.assert_array_equal(result.bias.isel(model=1), np.full((3, 4), -3))
    assert result.mappables[0].norm is result.mappables[4].norm
    assert result.mappables[2].norm is result.mappables[5].norm
    np.testing.assert_array_equal(result.paired_counts, [12, 12])


def test_missing_and_infinite_pairs_mask_bias(fields):
    obs, model = fields
    obs = obs.copy()
    model = model.copy()
    obs[0, 0] = np.nan
    model[0, 1] = np.inf
    result = mp.compare_spatial(obs, model, coastlines=False)
    assert int(result.paired_counts) == 10
    assert np.isnan(result.bias[0, :2]).all()
    assert result.field_norm.vmax == 13


def test_labels_cannot_silently_intersect(fields):
    obs, model = fields
    with pytest.raises(ValueError, match="coordinate labels differ"):
        mp.compare_spatial(
            obs, model.assign_coords(lon=[-100, -90, -80, -70]), coastlines=False
        )


def test_auxiliary_curvilinear_coordinates_must_match(fields):
    obs, model = fields
    lon, lat = np.meshgrid(obs.lon, obs.lat)
    obs = obs.rename(lat="j", lon="i").assign_coords(
        longitude=(("j", "i"), lon), latitude=(("j", "i"), lat)
    )
    model = obs + 2
    model = model.assign_coords(longitude=model.longitude + 0.5)
    with pytest.raises(ValueError, match="spatial coordinate"):
        mp.compare_spatial(obs, model, x="longitude", y="latitude", coastlines=False)


def test_station_pairing_and_bias_sign():
    obs = xr.DataArray(
        [1.0, 2.0, 3.0],
        dims="site",
        coords={
            "site": ["A", "B", "C"],
            "lon": ("site", [-100, -90, -80]),
            "lat": ("site", [30, 40, 35]),
        },
    )
    result = mp.compare_spatial(obs, obs - 2, kind="scatter", coastlines=False)
    np.testing.assert_equal(result.mappables[2].get_array(), [-2, -2, -2])
    np.testing.assert_equal(
        result.mappables[0].get_offsets(), [[-100, 30], [-90, 40], [-80, 35]]
    )


@pytest.mark.parametrize("bad_units", ["K", None])
def test_units_are_explicit_and_compatible(fields, bad_units):
    obs, model = fields
    model.attrs = {} if bad_units is None else {"units": bad_units}
    with pytest.raises(ValueError, match="units differ"):
        mp.compare_spatial(obs, model)


def test_station_grid_broadcast_rejected(fields):
    obs, model = fields
    with pytest.raises(ValueError, match="field dimensions differ"):
        mp.compare_spatial(
            obs.rename(lat="site").assign_coords(lat=("site", obs.lat.values)), model
        )


def test_initialization_is_lazy_and_explicit_scales_skip_global_reductions(fields):
    obs, model = fields
    obs, model = obs.chunk({"lat": 1}), model.chunk({"lat": 1})
    before = obs.copy(deep=True)
    executed = []
    with Callback(posttask=lambda *args: executed.append(args)):
        result = mp.SpatialComparisonPlot(obs, model, coastlines=False)
    assert executed == []
    # No full-field min/max computation is scheduled for explicit scales.
    keys = []
    with Callback(posttask=lambda key, *args: keys.append(str(key))):
        result.plot(vmin=0, vmax=20, bias_limit=5)
    assert not any("nanmin" in k or "nanmax" in k for k in keys)
    xr.testing.assert_identical(obs, before)
    assert hasattr(result.bias.data, "dask")


def test_auto_reductions_share_one_compute_graph(fields, monkeypatch):
    import monet_plots.plots.spatial_comparison as module

    obs, model = fields
    calls = []
    original = module.dask.compute

    def tracked(*args, **kwargs):
        calls.append(len(args))
        return original(*args, **kwargs)

    monkeypatch.setattr(module.dask, "compute", tracked)
    mp.compare_spatial(obs, model, coastlines=False)
    assert calls == [5]  # Four field bounds and one absolute bias maximum.


def test_user_norms_and_limits(fields):
    obs, model = fields
    norm = Normalize(0, 100)
    bias_norm = TwoSlopeNorm(0, -10, 5)
    result = mp.compare_spatial(
        obs, model, norm=norm, bias_norm=bias_norm, colorbar=False, coastlines=False
    )
    assert result.field_norm is norm and result.bias_norm is bias_norm
    assert result.colorbars == []
    with pytest.raises(ValueError, match="centered on zero"):
        mp.compare_spatial(obs, model, bias_norm=Normalize(-5, 5), coastlines=False)
    with pytest.raises(ValueError, match="finite and positive"):
        mp.compare_spatial(obs, model, bias_limit=0, coastlines=False)


def test_all_zero_bias_and_empty_row(fields):
    obs, _ = fields
    result = mp.compare_spatial(obs, obs, coastlines=False)
    assert result.bias_norm.vmin == -1 and result.bias_norm.vmax == 1
    empty = obs * np.nan
    with pytest.raises(ValueError, match="No finite paired values"):
        mp.compare_spatial(obs, empty, coastlines=False)
    result = mp.compare_spatial(
        obs, empty, vmin=0, vmax=20, bias_limit=2, coastlines=False
    )
    assert int(result.paired_counts) == 0


def test_export_and_failed_render_cleanup(fields, tmp_path):
    obs, model = fields
    before = len(plt.get_fignums())
    with pytest.raises(ValueError, match="bias_limit"):
        mp.compare_spatial(obs, model, bias_limit=-2, coastlines=False)
    assert len(plt.get_fignums()) == before
    result = mp.compare_spatial(obs, model, coastlines=False)
    for suffix in ["png", "pdf"]:
        path = tmp_path / f"comparison.{suffix}"
        result.save(path, dpi=80)
        assert path.stat().st_size > 1000


def test_row_materialization_reuses_dask_dependencies(fields):
    import dask.array as da
    from dask import delayed

    obs, _ = fields
    source_calls = []

    @delayed(pure=True)
    def source():
        source_calls.append(1)
        return obs.values

    lazy_obs = obs.copy(
        data=da.from_delayed(source(), shape=obs.shape, dtype=obs.dtype)
    )
    evaluations = {}
    for mode in ["panel", "row"]:
        source_calls.clear()
        result = mp.compare_spatial(
            lazy_obs,
            lazy_obs + 2,
            vmin=0,
            vmax=20,
            bias_limit=5,
            materialize=mode,
            coastlines=False,
        )
        evaluations[mode] = len(source_calls)
        np.testing.assert_allclose(result.mappables[2].get_array(), 2)
        result.close()
    # Task counts can rise as fusion changes. The costly shared source itself
    # must execute once per row, rather than once per individual panel.
    assert evaluations == {"panel": 3, "row": 1}


def test_coordinate_names_and_extra_dimensions_are_rejected(fields):
    obs, model = fields
    with pytest.raises(ValueError, match="same spatial coordinate names"):
        mp.compare_spatial(obs.rename(lon="longitude"), model, coastlines=False)
    with pytest.raises(ValueError, match="select extra dimensions"):
        mp.compare_spatial(
            obs.expand_dims(time=[0, 1]),
            model.expand_dims(time=[0, 1]),
            coastlines=False,
        )
    with pytest.raises(TypeError, match="DataArrays"):
        mp.compare_spatial(obs.to_dataset(), model)
