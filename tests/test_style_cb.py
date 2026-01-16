from monet_plots.style import CB_COLORS, CB_MARKERS

def test_cb_constants():
    assert len(CB_COLORS) == 8
    assert CB_COLORS[0] == "#000000"

    assert len(CB_MARKERS) >= 10
    assert "o" in CB_MARKERS
    assert "s" in CB_MARKERS
