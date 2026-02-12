import numpy as np
import xarray as xr
import dask.array as da
import pytest
from monet_plots import verification_metrics
from monet_plots.plots.reliability_diagram import ReliabilityDiagramPlot
import matplotlib.pyplot as plt

def test_reliability_curve_multidim_lazy():
    """Test reliability curve with multidimensional lazy inputs."""
    shape = (5, 5, 100) # lat, lon, time
    f_data = np.random.rand(*shape)
    o_data = np.random.randint(0, 2, shape)

    f_xr = xr.DataArray(da.from_array(f_data, chunks=(5, 5, 50)), dims=["lat", "lon", "time"])
    o_xr = xr.DataArray(da.from_array(o_data, chunks=(5, 5, 50)), dims=["lat", "lon", "time"])

    # Compute over time dimension
    bc, of, ct = verification_metrics.compute_reliability_curve(f_xr, o_xr, n_bins=5, dim="time")

    assert of.chunks is not None
    assert of.dims == ("lat", "lon", "bin")
    assert of.shape == (5, 5, 5)

    # Verify a single pixel
    pixel_f = f_data[0, 0, :]
    pixel_o = o_data[0, 0, :]
    bc_exp, of_exp, ct_exp = verification_metrics.compute_reliability_curve(pixel_f, pixel_o, n_bins=5)

    np.testing.assert_allclose(bc.values, bc_exp)
    np.testing.assert_allclose(of.isel(lat=0, lon=0).compute().values, of_exp)
    np.testing.assert_allclose(ct.isel(lat=0, lon=0).compute().values, ct_exp)

def test_brier_components_multidim_lazy():
    """Test Brier score components with multidimensional lazy inputs."""
    shape = (5, 5, 100)
    f_data = np.random.rand(*shape)
    o_data = np.random.randint(0, 2, shape)

    f_xr = xr.DataArray(da.from_array(f_data, chunks=(5, 5, 50)), dims=["lat", "lon", "time"])
    o_xr = xr.DataArray(da.from_array(o_data, chunks=(5, 5, 50)), dims=["lat", "lon", "time"])

    res = verification_metrics.compute_brier_score_components(f_xr, o_xr, n_bins=5, dim="time")

    assert res["reliability"].dims == ("lat", "lon")
    assert res["reliability"].chunks is not None

    # Verify one pixel
    pixel_f = f_data[0, 0, :]
    pixel_o = o_data[0, 0, :]
    res_exp = verification_metrics.compute_brier_score_components(pixel_f, pixel_o, n_bins=5)

    np.testing.assert_allclose(res["reliability"].isel(lat=0, lon=0).compute(), res_exp["reliability"])
    np.testing.assert_allclose(res["brier_score"].isel(lat=0, lon=0).compute(), res_exp["brier_score"])

def test_reliability_diagram_plot_xarray():
    """Test ReliabilityDiagramPlot with Xarray input."""
    ds = xr.Dataset({
        "forecast": (["time"], np.random.rand(100)),
        "observation": (["time"], np.random.randint(0, 2, 100))
    })

    plot = ReliabilityDiagramPlot()
    # This should now work without calling to_dataframe on the whole dataset
    ax = plot.plot(ds, forecasts_col="forecast", observations_col="observation", n_bins=5)

    assert isinstance(ax, plt.Axes)
    plt.close()

def test_reliability_diagram_plot_grouped_xarray():
    """Test ReliabilityDiagramPlot with grouped Xarray input."""
    ds = xr.Dataset({
        "forecast": (["time"], np.random.rand(100)),
        "observation": (["time"], np.random.randint(0, 2, 100)),
        "model": (["time"], ["A"]*50 + ["B"]*50)
    })

    plot = ReliabilityDiagramPlot()
    ax = plot.plot(ds, forecasts_col="forecast", observations_col="observation", label_col="model", n_bins=5)

    assert isinstance(ax, plt.Axes)
    plt.close()
