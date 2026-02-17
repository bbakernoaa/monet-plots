import numpy as np
import xarray as xr
from monet_plots.plots import GroupedDistributionPlot


def test_grouped_distribution_init():
    groups = ["A", "B", "C"]
    da_model = xr.DataArray(
        np.random.rand(3, 100),
        coords={"group": groups},
        dims=("group", "point"),
        name="AOD",
    )
    da_ref = xr.DataArray(
        np.random.rand(3, 100) * 0.8,
        coords={"group": groups},
        dims=("group", "point"),
        name="AOD",
    )

    plot = GroupedDistributionPlot(
        [da_model, da_ref], labels=["P8", "MERRA2"], group_dim="group"
    )
    assert plot.df_plot is None  # Data preparation is deferred
    plot._prepare_data()
    assert plot.df_plot is not None
    assert len(plot.df_plot) == 600  # 3 groups * 100 points * 2 models
    assert "group" in plot.df_plot.columns
    assert "value" in plot.df_plot.columns
    assert "Model" in plot.df_plot.columns

    # Test provenance
    assert "monet-plots.GroupedDistributionPlot" in da_model.attrs["history"]


def test_grouped_distribution_plot():
    groups = ["A", "B"]
    da_model = xr.DataArray(
        np.random.rand(2, 10),
        coords={"group": groups},
        dims=("group", "point"),
        name="AOD",
    )
    da_ref = xr.DataArray(
        np.random.rand(2, 10),
        coords={"group": groups},
        dims=("group", "point"),
        name="AOD",
    )

    plot = GroupedDistributionPlot(
        [da_model, da_ref], labels=["P8", "MERRA2"], group_dim="group"
    )
    ax = plot.plot()
    assert ax is not None
    assert ax == plot.ax
    assert plot.df_plot is not None
    assert "Group Distribution of AOD" in ax.get_title()


def test_grouped_distribution_three_datasets():
    groups = ["A", "B"]
    da1 = xr.DataArray(
        np.random.rand(2, 10),
        coords={"group": groups},
        dims=("group", "point"),
        name="VAR",
    )
    da2 = xr.DataArray(
        np.random.rand(2, 10),
        coords={"group": groups},
        dims=("group", "point"),
        name="VAR",
    )
    da3 = xr.DataArray(
        np.random.rand(2, 10),
        coords={"group": groups},
        dims=("group", "point"),
        name="VAR",
    )

    plot = GroupedDistributionPlot(
        [da1, da2, da3], labels=["M1", "M2", "M3"], group_dim="group"
    )
    plot._prepare_data()
    assert len(plot.df_plot["Model"].unique()) == 3
    ax = plot.plot()
    assert ax is not None
    assert "M1" in ax.get_legend().get_texts()[0].get_text()
    assert "Group Distribution of VAR: M1, M2 and M3" in ax.get_title()


def test_grouped_distribution_auto_labels():
    groups = ["A", "B"]
    da1 = xr.DataArray(
        np.random.rand(2, 10),
        coords={"group": groups},
        dims=("group", "point"),
        name="DATA1",
    )
    da2 = xr.DataArray(
        np.random.rand(2, 10),
        coords={"group": groups},
        dims=("group", "point"),
        name="DATA2",
    )

    plot = GroupedDistributionPlot([da1, da2], group_dim="group")
    plot.plot()
    assert "DATA1" in plot.labels
    assert "DATA2" in plot.labels
    assert plot.var_label == "DATA1"


def test_grouped_distribution_dask():
    import dask.array as da

    groups = ["A", "B"]
    da_model = xr.DataArray(
        da.random.random((2, 10), chunks=(1, 10)),
        coords={"group": groups},
        dims=("group", "point"),
        name="AOD",
    )
    da_ref = xr.DataArray(
        da.random.random((2, 10), chunks=(1, 10)),
        coords={"group": groups},
        dims=("group", "point"),
        name="AOD",
    )

    plot = GroupedDistributionPlot(
        [da_model, da_ref], labels=["P8", "MERRA2"], group_dim="group"
    )
    plot._prepare_data()
    assert plot.df_plot is not None
    assert len(plot.df_plot) == 40


def test_grouped_distribution_hvplot():
    groups = ["A", "B"]
    da_model = xr.DataArray(
        np.random.rand(2, 10),
        coords={"group": groups},
        dims=("group", "point"),
        name="AOD",
    )
    da_ref = xr.DataArray(
        np.random.rand(2, 10),
        coords={"group": groups},
        dims=("group", "point"),
        name="AOD",
    )

    plot = GroupedDistributionPlot(
        [da_model, da_ref], labels=["P8", "MERRA2"], group_dim="group"
    )
    hv_obj = plot.hvplot()
    assert hv_obj is not None
    assert plot.df_plot is not None
