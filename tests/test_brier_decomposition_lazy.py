import numpy as np
import xarray as xr
import dask.array as da
from monet_plots.plots.brier_decomposition import BrierScoreDecompositionPlot


def test_brier_decomposition_lazy_parity():
    # 1. Create Data
    n_samples = 200
    forecasts = np.random.rand(n_samples)
    observations = (np.random.rand(n_samples) > 0.5).astype(int)

    ds_eager = xr.Dataset(
        {"fcst": (("sample"), forecasts), "obs": (("sample"), observations)}
    )

    ds_lazy = xr.Dataset(
        {
            "fcst": (("sample"), da.from_array(forecasts, chunks=50)),
            "obs": (("sample"), da.from_array(observations, chunks=50)),
        }
    )

    # 2. Plot Eager
    plot_eager = BrierScoreDecompositionPlot()
    plot_eager.plot(ds_eager, forecasts_col="fcst", observations_col="obs")

    # 3. Plot Lazy
    plot_lazy = BrierScoreDecompositionPlot()
    plot_lazy.plot(ds_lazy, forecasts_col="fcst", observations_col="obs")

    # 4. Compare Bar Heights
    # Bars are Reliability, Resolution (-), Uncertainty
    # ax.containers[0] is reliability, [1] is resolution, [2] is uncertainty
    for i in range(3):
        h_eager = [b.get_height() for b in plot_eager.ax.containers[i]]
        h_lazy = [b.get_height() for b in plot_lazy.ax.containers[i]]
        np.testing.assert_allclose(h_eager, h_lazy)

    assert "Generated BrierScoreDecompositionPlot" in ds_lazy.attrs["history"]


def test_brier_decomposition_grouping_lazy():
    n_samples = 200
    forecasts = np.random.rand(n_samples)
    observations = (np.random.rand(n_samples) > 0.5).astype(int)
    labels = np.repeat(["A", "B"], n_samples // 2)

    ds_lazy = xr.Dataset(
        {
            "fcst": (("sample"), da.from_array(forecasts, chunks=50)),
            "obs": (("sample"), da.from_array(observations, chunks=50)),
            "model": (("sample"), labels),
        }
    )

    plot = BrierScoreDecompositionPlot()
    plot.plot(ds_lazy, forecasts_col="fcst", observations_col="obs", label_col="model")

    # Check that we have bars for two models
    # Each container should have 2 bars
    for i in range(3):
        assert len(plot.ax.containers[i]) == 2
