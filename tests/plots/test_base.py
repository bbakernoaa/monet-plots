import matplotlib.pyplot as plt
from monet_plots.plots.base import BasePlot


def test_baseplot_creates_fig_ax():
    plot = BasePlot()
    assert plot.fig is not None
    assert plot.ax is not None
    plt.close(plot.fig)


def test_baseplot_uses_passed_fig_ax():
    fig, ax = plt.subplots()
    plot = BasePlot(fig=fig, ax=ax)
    assert plot.fig is fig
    assert plot.ax is ax
    plt.close(fig)


def test_baseplot_fig_only():
    fig = plt.figure()
    plot = BasePlot(fig=fig)
    assert plot.fig is fig
    assert plot.ax is None  # New behavior: ax is None if only fig is provided
    plt.close(fig)


def test_baseplot_save(tmp_path):
    plot = BasePlot()
    filename = tmp_path / "test_plot.png"
    plot.save(str(filename))
    assert filename.exists()
    plt.close(plot.fig)


def test_baseplot_close():
    plot = BasePlot()
    fig = plot.fig
    plot.close()
    # After closing, figure should not be in the list of open figures
    assert fig.number not in plt.get_fignums()
