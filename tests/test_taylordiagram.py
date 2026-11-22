from monet_plots.taylordiagram import TaylorDiagram


def test_taylor_diagram_init():
    """Test the TaylorDiagram __init__ method."""
    dia = TaylorDiagram(1.0)
    assert dia is not None
    assert dia.refstd == 1.0
    assert dia.smax == 1.5
    assert dia.samplePoints[0].get_label() == "_"


def test_taylor_diagram_init_scale_label():
    """Test the TaylorDiagram __init__ method with scale and label."""
    dia = TaylorDiagram(1.0, scale=2.0, label="Reference")
    assert dia.smax == 2.0
    assert dia.samplePoints[0].get_label() == "Reference"


def test_taylor_diagram_add_sample():
    """Test the TaylorDiagram add_sample method."""
    dia = TaylorDiagram(1.0)
    dia.add_sample(1.1, 0.9)
    assert len(dia.samplePoints) == 2


def test_taylor_diagram_add_contours():
    """Test the TaylorDiagram add_contours method."""
    dia = TaylorDiagram(1.0)
    contours = dia.add_contours()
    assert contours is not None


def test_samples_expand_rendered_domain(tmp_path):
    import numpy as np
    from matplotlib import pyplot as plt

    dia = TaylorDiagram(1.0)
    first = dia.add_sample(1.2, 0.8, label="First")
    dia.add_contours(levels=[1.0, 5.0], colors="0.5")
    fig = dia.ax.figure
    try:
        fig.canvas.draw()
        last = dia.add_sample(10.0, -1.0, label="Outlier")
        fig.canvas.draw()
        assert dia.samplePoints[1] is first
        assert dia.smax > 10.0
        np.testing.assert_allclose(last.get_data(), [[np.pi], [10.0]])
        center = last.get_transform().transform((np.pi, 10.0))
        assert dia._ax.bbox.contains(*center)
        assert not last.get_clip_on()
        for suffix in ("png", "pdf"):
            fig.savefig(tmp_path / f"taylor.{suffix}")
    finally:
        plt.close(fig)
