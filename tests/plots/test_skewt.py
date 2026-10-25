import matplotlib.pyplot as plt
import numpy as np
import pytest

from monet_plots import SkewTPlot


def test_sounding_units_and_subplot(tmp_path):
    fig, ax = plt.subplots(subplot_kw={"projection": "monet_skewt"})
    plot = SkewTPlot(ax=ax, style=None)
    plot.plot(
        [100000, 85000, 50000],
        [298.15, 288.15, 253.15],
        pressure_units="Pa",
        temperature_units="K",
        label="Observed",
    )
    np.testing.assert_allclose(plot.overlays[0].get_xdata(), [25, 15, -20])
    assert ax.yaxis_inverted()
    plot.save(tmp_path / "sounding.pdf")
    plot.close()


def test_invalid_pressure():
    plot = SkewTPlot(style=None)
    try:
        with pytest.raises(ValueError, match="positive pressures"):
            plot.plot([1000, 0], [20, 10])
    finally:
        plot.close()


def test_moist_adiabat_is_finite_and_cools():
    from monet_plots.plots.skewt import _moist

    pressure = np.geomspace(900, 300, 20)
    temperature = _moist(pressure, 290.0)
    assert np.all(np.isfinite(temperature))
    assert np.all(np.diff(temperature) < 0)


def test_height_axis_can_be_forced():
    from monet_plots.plots.vertical import configure_vertical_axis

    plot = SkewTPlot(style=None)
    try:
        plot.ax.invert_yaxis()
        configure_vertical_axis(plot.ax, np.array([1, 2]), mode="height")
        assert not plot.ax.yaxis_inverted()
    finally:
        plot.close()
