import dask.array as da
import numpy as np
import xarray as xr

from monet_plots.plots.rank_histogram import RankHistogramPlot


def test_rank_histogram_lazy_parity():
    # 1. Create Eager Data
    n_samples = 100
    n_members = 10
    ranks = np.random.randint(0, n_members + 1, n_samples)

    da_eager = xr.DataArray(ranks, dims="sample", name="rank")

    # 2. Create Lazy Data
    da_lazy = xr.DataArray(da.from_array(ranks, chunks=25), dims="sample", name="rank")

    # 3. Plot Eager
    plot_eager = RankHistogramPlot()
    plot_eager.plot(da_eager, n_members=n_members, normalize=True)

    # 4. Plot Lazy
    plot_lazy = RankHistogramPlot()
    plot_lazy.plot(da_lazy, n_members=n_members, normalize=True)

    # 5. Compare Bar Heights (the first axes children are the bars)
    bars_eager = [b.get_height() for b in plot_eager.ax.containers[0]]
    bars_lazy = [b.get_height() for b in plot_lazy.ax.containers[0]]

    np.testing.assert_allclose(bars_eager, bars_lazy)
    assert "Generated RankHistogramPlot" in da_lazy.attrs["history"]


def test_rank_histogram_grouping_lazy():
    n_samples = 100
    n_members = 5
    ranks = np.random.randint(0, n_members + 1, n_samples)
    labels = np.repeat(["A", "B"], n_samples // 2)

    ds_lazy = xr.Dataset(
        {
            "rank": (("sample"), da.from_array(ranks, chunks=25)),
            "label": (("sample"), da.from_array(labels, chunks=25)),
        }
    )

    plot = RankHistogramPlot()
    plot.plot(ds_lazy, rank_col="rank", label_col="label", n_members=n_members)

    # Check that we have two sets of bars (two containers)
    assert len(plot.ax.containers) == 2
    assert "Generated RankHistogramPlot" in ds_lazy.attrs["history"]
