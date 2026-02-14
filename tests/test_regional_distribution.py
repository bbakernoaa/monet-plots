import numpy as np
import xarray as xr
from monet_plots.plots import RegionalDistributionPlot


def test_regional_distribution_init():
    regions = ["WAF", "SAF", "SAH", "AMZ", "SEA"]
    da_model = xr.DataArray(
        np.random.rand(5, 100),
        coords={"region": regions},
        dims=("region", "point"),
        name="AOD",
    )
    da_ref = xr.DataArray(
        np.random.rand(5, 100) * 0.8,
        coords={"region": regions},
        dims=("region", "point"),
        name="AOD",
    )

    plot = RegionalDistributionPlot(
        [da_model, da_ref], labels=["P8", "MERRA2"], group_dim="region"
    )
    assert plot.df_plot is None  # Data preparation is deferred
    plot._prepare_data()
    assert plot.df_plot is not None
    assert len(plot.df_plot) == 1000  # 5 regions * 100 points * 2 models
    assert "region" in plot.df_plot.columns
    assert "value" in plot.df_plot.columns
    assert "Model" in plot.df_plot.columns

    # Test provenance
    assert "monet-plots.RegionalDistributionPlot" in da_model.attrs["history"]


def test_regional_distribution_plot():
    regions = ["WAF", "SAF"]
    da_model = xr.DataArray(
        np.random.rand(2, 10),
        coords={"region": regions},
        dims=("region", "point"),
        name="AOD",
    )
    da_ref = xr.DataArray(
        np.random.rand(2, 10),
        coords={"region": regions},
        dims=("region", "point"),
        name="AOD",
    )

    plot = RegionalDistributionPlot(
        [da_model, da_ref], labels=["P8", "MERRA2"], group_dim="region"
    )
    ax = plot.plot()
    assert ax is not None
    assert ax == plot.ax
    assert plot.df_plot is not None


def test_regional_distribution_inset_map():
    regions = ["WAF", "SAF"]
    da_model = xr.DataArray(
        np.random.rand(2, 10),
        coords={"region": regions},
        dims=("region", "point"),
        name="AOD",
    )
    da_ref = xr.DataArray(
        np.random.rand(2, 10),
        coords={"region": regions},
        dims=("region", "point"),
        name="AOD",
    )

    plot = RegionalDistributionPlot(
        [da_model, da_ref], labels=["P8", "MERRA2"], group_dim="region"
    )
    ax_inset = plot.add_inset_map()
    assert ax_inset is not None
    assert ax_inset != plot.ax


def test_regional_distribution_dask():
    import dask.array as da

    regions = ["WAF", "SAF"]
    da_model = xr.DataArray(
        da.random.random((2, 10), chunks=(1, 10)),
        coords={"region": regions},
        dims=("region", "point"),
        name="AOD",
    )
    da_ref = xr.DataArray(
        da.random.random((2, 10), chunks=(1, 10)),
        coords={"region": regions},
        dims=("region", "point"),
        name="AOD",
    )

    plot = RegionalDistributionPlot(
        [da_model, da_ref], labels=["P8", "MERRA2"], group_dim="region"
    )
    plot._prepare_data()
    assert plot.df_plot is not None
    assert len(plot.df_plot) == 40


def test_regional_distribution_hvplot():
    regions = ["WAF", "SAF"]
    da_model = xr.DataArray(
        np.random.rand(2, 10),
        coords={"region": regions},
        dims=("region", "point"),
        name="AOD",
    )
    da_ref = xr.DataArray(
        np.random.rand(2, 10),
        coords={"region": regions},
        dims=("region", "point"),
        name="AOD",
    )

    plot = RegionalDistributionPlot(
        [da_model, da_ref], labels=["P8", "MERRA2"], group_dim="region"
    )
    hv_obj = plot.hvplot()
    assert hv_obj is not None
    assert plot.df_plot is not None
