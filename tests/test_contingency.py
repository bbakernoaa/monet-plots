import numpy as np
import pytest
import xarray as xr

try:
    import dask.array as da
except ImportError:
    da = None

from monet_plots.plots.performance_diagram import PerformanceDiagramPlot
from monet_plots.verification_metrics import (
    compute_categorical_metrics,
    compute_contingency_table,
)


def test_contingency_table_parity():
    obs_np = np.array([1.0, 0.0, 1.0, 2.0, 0.5])
    mod_np = np.array([0.8, 0.2, 1.2, 1.5, 0.1])
    threshold = 0.5

    # NumPy
    res_np = compute_contingency_table(obs_np, mod_np, threshold)

    # Xarray
    obs_xr = xr.DataArray(obs_np, dims="time")
    mod_xr = xr.DataArray(mod_np, dims="time")
    res_xr = compute_contingency_table(obs_xr, mod_xr, threshold)

    for key in ["hits", "misses", "fa", "cn"]:
        np.testing.assert_array_equal(res_np[key], res_xr[key].values)

    # Dask (if available)
    if da is not None:
        obs_da = xr.DataArray(da.from_array(obs_np, chunks=2), dims="time")
        mod_da = xr.DataArray(da.from_array(mod_np, chunks=2), dims="time")
        res_da = compute_contingency_table(obs_da, mod_da, threshold)

        for key in ["hits", "misses", "fa", "cn"]:
            np.testing.assert_array_equal(res_np[key], res_da[key].values)
            assert isinstance(res_da[key].data, da.Array)


def test_categorical_metrics_lazy():
    obs_np = np.array([1, 0, 1, 1])
    mod_np = np.array([1, 1, 0, 1])
    threshold = 0.5

    obs_xr = xr.DataArray(obs_np, dims="time")
    mod_xr = xr.DataArray(mod_np, dims="time")

    if da is not None:
        obs_xr = obs_xr.chunk({"time": 2})
        mod_xr = mod_xr.chunk({"time": 2})

    metrics = compute_categorical_metrics(obs_xr, mod_xr, threshold, dim="time")

    # POD = hits / (hits + misses). hits=(obs>=0.5 & mod>=0.5) = [T, F, F, T].sum()=2
    # misses=(obs>=0.5 & mod<0.5) = [F, F, T, F].sum()=1
    # fa=(obs<0.5 & mod>=0.5) = [F, T, F, F].sum()=1
    # POD = 2 / (2 + 1) = 2/3
    # FAR = 1 / (2 + 1) = 1/3
    assert float(metrics["pod"]) == pytest.approx(2 / 3)
    assert float(metrics["far"]) == pytest.approx(1 / 3)
    assert "history" in metrics["pod"].attrs


def test_performance_diagram_lazy():
    if da is None:
        pytest.skip("Dask required for this test")

    obs = xr.DataArray(
        da.random.random((10, 5), chunks=(5, 5)), dims=("time", "site"), name="obs"
    )
    mod = xr.DataArray(
        da.random.random((10, 5), chunks=(5, 5)), dims=("time", "site"), name="mod"
    )
    ds = xr.Dataset({"obs": obs, "mod": mod})

    plot = PerformanceDiagramPlot()
    # Test plotting with threshold and dim reduction
    # This will trigger computation only during ax.plot calls
    ax = plot.plot(
        ds, obs_col="obs", mod_col="mod", threshold=0.5, dim="time", label_col="site"
    )

    assert len(ax.get_legend().get_texts()) == 5
    # Check if history was updated
    assert "PerformanceDiagramPlot" in ds.attrs["history"]


def test_performance_diagram_counts_xarray():
    # Test with pre-calculated counts in Xarray
    hits = xr.DataArray([10, 20], dims="site", coords={"site": ["A", "B"]})
    misses = xr.DataArray([5, 2], dims="site")
    fa = xr.DataArray([2, 5], dims="site")
    cn = xr.DataArray([100, 100], dims="site")

    ds = xr.Dataset({"hits": hits, "misses": misses, "fa": fa, "cn": cn})

    plot = PerformanceDiagramPlot()
    ax = plot.plot(ds, counts_cols=["hits", "misses", "fa", "cn"], label_col="site")

    assert len(ax.get_legend().get_texts()) == 2
