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


def test_journal_style_is_scoped_and_preserves_export_width(tmp_path):
    from matplotlib import pyplot as plt

    before = plt.rcParams.copy()
    plot = BasePlot(style="journal", figsize=(3.2, 2.5))
    try:
        assert plt.rcParams == before
        output = tmp_path / "figure.pdf"
        plot.save(output, dpi=200)
        assert b"/MediaBox [ 0 0 230.4 180 ]" in output.read_bytes()
        assert plt.rcParams == before
    finally:
        plot.close()


def test_save_wraps_each_axes_title_to_its_own_width(tmp_path, monkeypatch):
    fig, axes = plt.subplots(
        1, 2, figsize=(6, 3), width_ratios=[1, 2], layout="constrained"
    )
    plot = BasePlot(fig=fig)
    text = "Observed and forecast temperature across the study region"
    narrow = axes[0].set_title(text, loc="left")
    wide = axes[1].set_title("Temperature\nJuly 2026")
    savefig = fig.savefig

    def check_titles(*args, **kwargs):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        assert "\n" in narrow.get_text()
        assert " ".join(narrow.get_text().split()) == text
        assert narrow.get_window_extent(renderer).width <= axes[0].bbox.width
        assert wide.get_text() == "Temperature\nJuly 2026"
        return savefig(*args, **kwargs)

    monkeypatch.setattr(fig, "savefig", check_titles)
    try:
        plot.save(tmp_path / "wrapped.png")
        plot.save(tmp_path / "wrapped.pdf", bbox_inches="tight")
        assert narrow.get_text() == text
        assert (tmp_path / "wrapped.pdf").stat().st_size
    finally:
        plot.close()


def test_save_reserves_vertical_space_for_wrapped_title(tmp_path, monkeypatch):
    plot = BasePlot(style="journal", figsize=(2.2, 1.5))
    text = "Comparing observed and forecast temperatures over the verification period"
    title = plot.ax.set_title(text)
    position = plot.ax.get_position(original=True).bounds
    savefig = plot.fig.savefig

    def check_bounds(*args, **kwargs):
        plot.fig.canvas.draw()
        bounds = title.get_window_extent(plot.fig.canvas.get_renderer())
        assert bounds.y1 <= plot.fig.bbox.y1
        return savefig(*args, **kwargs)

    monkeypatch.setattr(plot.fig, "savefig", check_bounds)
    try:
        plot.save(tmp_path / "narrow.png")
        plot.save(tmp_path / "narrow.pdf")
        assert title.get_text() == text
        assert plot.ax.get_position(original=True).bounds == position
    finally:
        plot.close()
