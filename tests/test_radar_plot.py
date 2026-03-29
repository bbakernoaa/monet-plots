import numpy as np
import pytest
import xarray as xr
from monet_plots.plots.radar import RadarPlot
from monet_plots.verification_metrics import compute_radar_metrics


@pytest.fixture
def sample_data():
    obs = np.array([1, 2, 3, 4, 5])
    mod1 = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
    mod2 = np.array([1.5, 2.5, 3.5, 4.5, 5.5])

    ds = xr.Dataset(
        {"obs": (["time"], obs), "mod1": (["time"], mod1), "mod2": (["time"], mod2)}
    )
    return ds


def test_compute_radar_metrics(sample_data):
    obs = sample_data.obs
    mod = sample_data.mod1

    ds = compute_radar_metrics(obs, mod)

    assert isinstance(ds, xr.Dataset)
    for var in ["R", "NMB", "NME", "RMSE", "MAE"]:
        assert var in ds.data_vars
        assert 0 <= ds[var].values <= 1


def test_radar_plot_init_data(sample_data):
    radar = RadarPlot(data=sample_data, obs_col="obs", mod_cols=["mod1", "mod2"])
    assert "model" in radar.metrics_data.dims
    assert len(radar.metrics_data.model) == 2
    assert "R" in radar.metrics_data.data_vars


def test_radar_plot_init_metrics_data(sample_data):
    metrics_ds = compute_radar_metrics(sample_data.obs, sample_data.mod1)
    metrics_ds = metrics_ds.expand_dims(model=["model1"])

    radar = RadarPlot(metrics_data=metrics_ds)
    assert "model1" in radar.metrics_data.model.values


def test_radar_plot_draw(sample_data):
    radar = RadarPlot(data=sample_data, obs_col="obs", mod_cols=["mod1", "mod2"])
    ax = radar.plot()
    assert ax.name == "polar"
    # One line per model
    assert len(ax.get_lines()) == 2
    radar.close()


def test_radar_plot_lazy(sample_data):
    # Convert to dask
    ds_lazy = sample_data.chunk({"time": 2})
    radar = RadarPlot(data=ds_lazy, obs_col="obs", mod_cols=["mod1", "mod2"])
    # compute_radar_metrics should handle lazy objects
    assert "R" in radar.metrics_data.data_vars
    ax = radar.plot()
    assert len(ax.get_lines()) == 2
    radar.close()
