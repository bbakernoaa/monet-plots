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
