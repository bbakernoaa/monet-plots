import numpy as np
import pytest
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

try:
    import dask.array as da
except ImportError:
    da = None

from monet_plots.plots.rank_histogram import RankHistogramPlot


def create_test_data(lazy=False):
    """Create test data for rank histogram."""
    # 100 samples, 5 ensemble members
    n_samples = 100
    n_members = 5
    ranks = np.random.randint(0, n_members + 1, n_samples)

    if lazy and da is not None:
        ranks_da = da.from_array(ranks, chunks=50)
    else:
        ranks_da = ranks

    ds = xr.Dataset(
        {"rank": (["index"], ranks_da)}, coords={"index": np.arange(n_samples)}
    )

    return ds


def test_rank_histogram_init():
    """Test initialization of RankHistogramPlot."""
    ds = create_test_data()
    plot = RankHistogramPlot(ds)
    assert plot.data is not None
    assert isinstance(plot.data, xr.Dataset)
    assert "rank" in plot.data


@pytest.mark.skipif(da is None, reason="dask not installed")
def test_rank_histogram_eager_vs_lazy_parity():
    """Verify RankHistogramPlot yields identical results for eager and lazy data."""
    # Create fixed data for comparison
    n_samples = 100
    n_members = 5
    ranks = np.random.randint(0, n_members + 1, n_samples)

    ds_eager = xr.Dataset(
        {"rank": (["index"], ranks)}, coords={"index": np.arange(n_samples)}
    )

    ds_lazy = xr.Dataset(
        {"rank": (["index"], da.from_array(ranks, chunks=50))},
        coords={"index": np.arange(n_samples)},
    )

    # Plot eager
    plot_eager = RankHistogramPlot(ds_eager)
    ax_eager = plot_eager.plot()
    bars_eager = [rect.get_height() for rect in ax_eager.containers[0]]
    plt.close(ax_eager.figure)

    # Plot lazy
    plot_lazy = RankHistogramPlot(ds_lazy)
    ax_lazy = plot_lazy.plot()
    bars_lazy = [rect.get_height() for rect in ax_lazy.containers[0]]
    plt.close(ax_lazy.figure)

    # Compare bar heights (frequencies)
    np.testing.assert_allclose(bars_eager, bars_lazy)


def test_rank_histogram_grouped():
    """Test grouped rank histogram."""
    n_samples = 100
    ranks = np.random.randint(0, 6, n_samples)
    labels = np.random.choice(["A", "B"], n_samples)

    df = pd.DataFrame({"rank": ranks, "model": labels})

    plot = RankHistogramPlot(df)
    ax = plot.plot(label_col="model")

    # We have one container for the bars, but it might have multiple sets if labeled
    # Actually Matplotlib bar creates one container per call to bar
    # In grouped, we call bar multiple times.
    assert len(ax.containers) >= 2  # At least 2 groups
    assert ax.get_legend() is not None
    plt.close(ax.figure)


def test_rank_histogram_vectorized():
    """Test plotting from raw ensemble/obs data."""
    n_samples = 50
    n_members = 4
    ensemble = np.random.rand(n_samples, n_members)
    obs = np.random.rand(n_samples)

    ds = xr.Dataset(
        {"ensemble": (["index", "member"], ensemble), "obs": (["index"], obs)},
        coords={"index": np.arange(n_samples), "member": np.arange(n_members)},
    )

    plot = RankHistogramPlot(ds)
    ax = plot.plot(ensemble_col="ensemble", obs_col="obs", member_dim="member")

    assert len(ax.containers) == 1
    # 4 members -> 5 bins
    assert len(ax.containers[0]) == 5
    plt.close(ax.figure)


def test_rank_histogram_hvplot():
    """Test Track B hvplot generation."""
    ds = create_test_data()
    plot = RankHistogramPlot(ds)
    hv_obj = plot.hvplot()
    assert hv_obj is not None
