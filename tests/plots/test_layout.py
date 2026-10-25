import pytest

from monet_plots import PlotLayout, SkewTPlot
from monet_plots.plots.profile import ProfilePlot


def test_layout_composes_different_plot_axes(tmp_path):
    layout = PlotLayout((1, 2), figsize=(9, 5))
    try:
        skew = layout.add(
            SkewTPlot,
            cell=(0, 0),
            name="sounding",
            subplot_kwargs={"projection": "monet_skewt"},
        )
        skew.plot([1000, 850, 500], [20, 10, -20], label="Observed")
        profile = layout.add(
            ProfilePlot,
            cell=(0, 1),
            name="profile",
            plot_kwargs={"x": [1, 2, 3], "y": [0, 1, 2], "style": None},
        )
        profile.plot()
        assert layout.axes["sounding"] is skew.ax
        assert layout.axes["profile"] is profile.ax
        layout.save(tmp_path / "layout.png")
    finally:
        layout.close()


def test_layout_rejects_out_of_bounds_cell():
    layout = PlotLayout((1, 1))
    try:
        with pytest.raises(ValueError, match="fit within"):
            layout.add(ProfilePlot, cell=(1, 0))
    finally:
        layout.close()
