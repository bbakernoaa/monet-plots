import numpy as np
import xarray as xr
from dask.callbacks import Callback

from monet_plots.plots.radar import RadarPlot


def test_radar_loads_shared_observations_once_across_models():
    data = xr.Dataset(
        {
            "obs": ("sample", [1.0, 4.0, 2.0, 5.0, np.nan, 8.0]),
            "a": ("sample", [2.0, 3.0, 2.0, 6.0, 4.0, 7.0]),
            "b": ("sample", [1.0, np.nan, 3.0, 5.0, 6.0, 9.0]),
        }
    )
    opts = {"obs_col": "obs", "mod_cols": ["a", "b"], "metrics": ["RMSE", "R", "NMB"]}
    expected = RadarPlot(data, **opts)
    actual = None
    try:
        runs = []
        with Callback(start=lambda graph: runs.append(len(graph))):
            actual = RadarPlot(data.chunk({"sample": 2}), **opts)
            actual.plot()
        assert (
            len(runs) == 3
        )  # One run per source array, including shared observations.
        xr.testing.assert_allclose(actual.metrics_data, expected.metrics_data)
    finally:
        expected.close()
        if actual is not None:
            actual.close()
