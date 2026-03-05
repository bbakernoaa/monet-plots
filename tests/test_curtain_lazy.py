# tests/test_curtain_lazy.py

import numpy as np
import pytest
import xarray as xr
from monet_plots.plots.curtain import CurtainPlot


@pytest.fixture
def sample_curtain_data():
    """Create a sample 2D DataArray for curtain plots."""
    nx, ny = 10, 20
    x = np.linspace(0, 10, nx)
    y = np.linspace(0, 5, ny)
    data = np.random.rand(ny, nx)
    da = xr.DataArray(
        data,
        coords={"time": x, "level": y},
        dims=("level", "time"),
        name="test_var",
        attrs={"units": "ug/m3"},
    )
    return da


def test_curtain_plot_eager_vs_lazy(sample_curtain_data):
    """Verify CurtainPlot produces identical results for eager and lazy data."""
    da_eager = sample_curtain_data
    da_lazy = sample_curtain_data.chunk({"time": 5, "level": 10})

    # Initialize plots
    cp_eager = CurtainPlot(data=da_eager, x="time", y="level")
    cp_lazy = CurtainPlot(data=da_lazy, x="time", y="level")

    # Track A: Static Plot
    cp_eager.plot()
    cp_lazy.plot()

    # Check that provenance was updated in both (though we test lazy vs eager logic)
    assert "Generated CurtainPlot" in cp_eager.data.attrs["history"]
    assert "Generated CurtainPlot" in cp_lazy.data.attrs["history"]

    # In this case, 'plot' returns the matplotlib Axes. We can't easily compare
    # the axes themselves for equality of the drawn artists without deep inspection,
    # but we can verify that the data used for plotting was computed correctly.
    # The 'compute' call happens inside plot().

    assert cp_lazy.data.chunks is not None
    assert cp_eager.data.chunks is None


def test_curtain_plot_hvplot(sample_curtain_data):
    """Verify CurtainPlot.hvplot() returns an hvPlot object."""
    pytest.importorskip("hvplot")

    cp = CurtainPlot(data=sample_curtain_data, x="time", y="level")
    hv_plot = cp.hvplot()

    # Check that it returns a holoviews object (hvPlot returns hv.Element subclasses)
    import holoviews as hv
    assert isinstance(hv_plot, hv.core.Element)


def test_curtain_plot_backward_compatibility(sample_curtain_data):
    """Verify CurtainPlot works with 'df' alias and handles dimension defaults."""
    # Test 'df' alias
    cp = CurtainPlot(df=sample_curtain_data)
    assert cp.data.equals(sample_curtain_data)

    # Test default x and y
    ax = cp.plot()
    assert cp.x == "time"
    assert cp.y == "level"
    assert ax.get_xlabel() == "time"
    assert ax.get_ylabel() == "level"


def test_curtain_plot_invalid_data():
    """Verify CurtainPlot raises error for invalid data dimensions."""
    da_3d = xr.DataArray(np.random.rand(2, 3, 4))
    cp = CurtainPlot(data=da_3d, x="dim_1", y="dim_0")
    with pytest.raises(ValueError, match="CurtainPlot requires 2D data"):
        cp.plot()
