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


def test_sounding_and_guides_reach_surface_pressure():
    plot = SkewTPlot(style=None)
    try:
        plot.plot([1000, 850, 500, 100], [24, 13, -18, -68])
        plot.fig.canvas.draw()
        bbox = plot.ax.bbox
        for line in plot.ax.lines:
            pressure = np.asarray(line.get_ydata())
            if pressure.size and pressure.max() >= 1000 and pressure.min() <= 100:
                vertices = line.get_transform().transform(
                    np.column_stack((line.get_xdata(), pressure))
                )
                visible = np.isfinite(vertices).all(axis=1)
                assert np.any(
                    visible
                    & (vertices[:, 1] <= bbox.ymax)
                    & (vertices[:, 1] >= bbox.ymin)
                )
        assert plot.ax.transData.transform((24, 1000))[0] < bbox.xmax
    finally:
        plot.close()


def test_skewt_default_aspect_is_meteorological():
    plot = SkewTPlot(style=None)
    try:
        assert plot.ax.get_aspect() == 80.5
        custom = SkewTPlot(aspect=60, style=None)
        try:
            assert custom.ax.get_aspect() == 60
        finally:
            custom.close()
    finally:
        plot.close()
