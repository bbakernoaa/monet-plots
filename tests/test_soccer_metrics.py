import numpy as np
import pandas as pd
import pytest
import xarray as xr

from monet_plots import verification_metrics
from monet_plots.plots.soccer import SoccerPlot

try:
    import dask.array as da
except ImportError:
    da = None

try:
    import holoviews as hv
    import hvplot  # noqa: F401
except ImportError:
    hv = None


def test_soccer_metrics_eager():
    """Test soccer metrics with NumPy (Eager)."""
    obs = np.array([10, 20, 30])
    mod = np.array([12, 18, 35])

    # MFB = mean(200 * (mod - obs) / (mod + obs))
    expected_fb_element = 200.0 * (mod - obs) / (mod + obs)
    expected_mfb = np.mean(expected_fb_element)

    mfb = verification_metrics.compute_mfb(obs, mod)
    np.testing.assert_allclose(mfb, expected_mfb)

    # MFE
    expected_mfe = np.mean(np.abs(expected_fb_element))
    mfe = verification_metrics.compute_mfe(obs, mod)
    np.testing.assert_allclose(mfe, expected_mfe)

    # NMB = 100 * sum(mod - obs) / sum(obs)
    expected_nmb = 100.0 * np.sum(mod - obs) / np.sum(obs)
    nmb = verification_metrics.compute_nmb(obs, mod)
    np.testing.assert_allclose(nmb, expected_nmb)

    # NME
    expected_nme = 100.0 * np.sum(np.abs(mod - obs)) / np.sum(obs)
    nme = verification_metrics.compute_nme(obs, mod)
    np.testing.assert_allclose(nme, expected_nme)


@pytest.mark.skipif(da is None, reason="dask not installed")
def test_soccer_metrics_lazy():
    """Test soccer metrics with Dask/Xarray (Lazy)."""
    obs_data = np.array([10, 20, 30, 40, 50])
    mod_data = np.array([12, 18, 35, 38, 55])

    obs = xr.DataArray(da.from_array(obs_data, chunks=2), dims=["x"])
    mod = xr.DataArray(da.from_array(mod_data, chunks=2), dims=["x"])

    mfb = verification_metrics.compute_mfb(obs, mod)
    assert mfb.chunks is not None
    assert "Calculated MFB" in mfb.attrs["history"]

    expected_fb = np.mean(200.0 * (mod_data - obs_data) / (mod_data + obs_data))
    np.testing.assert_allclose(mfb.compute(), expected_fb)

    # Test element-wise (dim=[])
    fb_element = verification_metrics.compute_mfb(obs, mod, dim=[])
    assert fb_element.chunks is not None
    assert fb_element.shape == (5,)
    expected_fb_element = 200.0 * (mod_data - obs_data) / (mod_data + obs_data)
    np.testing.assert_allclose(fb_element.compute(), expected_fb_element)


@pytest.mark.skipif(hv is None, reason="hvplot or holoviews not installed")
def test_soccer_hvplot():
    """Test SoccerPlot.hvplot() returns a valid object."""
    df = pd.DataFrame({"obs": [1, 2], "mod": [1.1, 1.9], "label": ["A", "B"]})
    plot = SoccerPlot(df, obs_col="obs", mod_col="mod", label_col="label")
    hv_obj = plot.hvplot()
    assert hv_obj is not None
    # Verify it's a holoviews object
    assert isinstance(hv_obj, hv.core.dimension.Dimensioned)


@pytest.mark.skipif(da is None, reason="dask not installed")
@pytest.mark.skipif(hv is None, reason="hvplot or holoviews not installed")
def test_soccer_hvplot_lazy():
    """Test SoccerPlot.hvplot() with lazy xarray."""
    obs_data = np.array([10, 20, 30])
    mod_data = np.array([12, 18, 35])

    ds = xr.Dataset(
        {
            "obs": (["x"], obs_data),
            "mod": (["x"], mod_data),
        },
        coords={"x": [0, 1, 2]},
    ).chunk({"x": 2})

    plot = SoccerPlot(ds, obs_col="obs", mod_col="mod")
    hv_obj = plot.hvplot()
    assert hv_obj is not None

    assert isinstance(hv_obj, hv.core.dimension.Dimensioned)
