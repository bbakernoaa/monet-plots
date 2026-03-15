import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from monet_plots.plots.spatial_overlay import SpatialOverlayPlot


def test_spatial_overlay_plot():
    """Test SpatialOverlayPlot with model grid and observation points."""
    # Create model data
    lon = np.linspace(-120, -70, 10)
    lat = np.linspace(25, 50, 10)
    data = np.random.rand(10, 10)
    model_da = xr.DataArray(
        data, coords={"lat": lat, "lon": lon}, dims=["lat", "lon"], name="model"
    )

    # Create observation data
    obs_lon = np.random.uniform(-120, -70, 20)
    obs_lat = np.random.uniform(25, 50, 20)
    obs_val = np.random.rand(20)
    obs_df = pd.DataFrame({"lat": obs_lat, "lon": obs_lon, "obs": obs_val})

    plot = SpatialOverlayPlot(model_da, obs_df, model_var="model", obs_var="obs")
    ax = plot.plot(kind="contourf", coastlines=True)

    assert ax is not None
    assert hasattr(ax, "projection")
    # Check for contourf (collection) and scatter (collection)
    assert len(ax.collections) >= 2
    plt.close(plot.fig)
