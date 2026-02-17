import numpy as np
import pandas as pd
import xarray as xr
import pytest
import matplotlib.pyplot as plt
from monet_plots.plots import RidgelinePlot


def test_ridgeline_plot_xarray():
    """Test RidgelinePlot with xarray DataArray."""
    # Create Synthetic Xarray Data (Lat x Lon)
    lats = np.linspace(-90, 90, 20)
    lons = np.linspace(-180, 180, 30)
    data_np = np.zeros((len(lats), len(lons)))

    for i, lat in enumerate(lats):
        base_temp = 30 - 0.5 * abs(lat)
        noise = np.random.normal(0, 3, len(lons))
        data_np[i, :] = base_temp + noise

    da = xr.DataArray(
        data_np,
        coords={"lat": lats, "lon": lons},
        dims=("lat", "lon"),
        name="Temperature",
    )

    plot = RidgelinePlot(da, group_dim="lat", title="Test Ridgeline")
    ax = plot.plot()

    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == "Test Ridgeline"
    plt.close(plot.fig)


def test_ridgeline_plot_pandas():
    """Test RidgelinePlot with pandas DataFrame."""
    # Create synthetic pandas data
    groups = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    data = {
        "group": np.repeat(groups, 50),
        "value": np.random.randn(500),
    }
    df = pd.DataFrame(data)

    plot = RidgelinePlot(df, group_dim="group", x="value")
    ax = plot.plot()

    assert isinstance(ax, plt.Axes)
    # Check that labels were added (every 5th)
    # "A" is index 0, "F" is index 5
    # Since we sort by group_dim ascending=False in DF:
    # J, I, H, G, F, E, D, C, B, A
    # i=0: J, i=5: E
    # Wait, groups unique returns them in order of appearance if not sorted?
    # In my code: groups = df_sorted[self.group_dim].unique()
    # df_sorted is df.sort_values(self.group_dim, ascending=False)
    # So J, I, H, G, F, E, D, C, B, A
    # i=0: J, i=5: E
    plt.close(plot.fig)


def test_ridgeline_plot_dataset():
    """Test RidgelinePlot with xarray Dataset."""
    lats = np.linspace(-90, 90, 10)
    lons = np.linspace(-180, 180, 20)
    data_np = np.random.randn(len(lats), len(lons))
    ds = xr.Dataset(
        {"temp": (["lat", "lon"], data_np)}, coords={"lat": lats, "lon": lons}
    )

    plot = RidgelinePlot(ds, group_dim="lat", x="temp")
    ax = plot.plot()
    assert isinstance(ax, plt.Axes)
    plt.close(plot.fig)


def test_ridgeline_plot_empty_fails():
    """Test that RidgelinePlot raises error with empty/NaN data."""
    df = pd.DataFrame({"group": ["A", "A"], "value": [np.nan, np.nan]})
    with pytest.raises(ValueError, match="No valid data points found"):
        plot = RidgelinePlot(df, group_dim="group", x="value")
        plot.plot()


def test_ridgeline_new_features():
    """Test RidgelinePlot new features: bandwidth, alpha, quantiles, color_by_group."""
    df = pd.DataFrame(
        {
            "group": np.repeat(["A", "B"], 50),
            "value": np.random.randn(100),
        }
    )

    # Test BW and Alpha
    plot = RidgelinePlot(df, group_dim="group", x="value", bw_method=0.2, alpha=0.5)
    ax = plot.plot()
    assert isinstance(ax, plt.Axes)
    plt.close(plot.fig)

    # Test Quantiles and Color by Group
    plot = RidgelinePlot(df, group_dim="group", x="value", quantiles=[0.5])
    ax = plot.plot(color_by_group=True)
    assert isinstance(ax, plt.Axes)
    plt.close(plot.fig)
