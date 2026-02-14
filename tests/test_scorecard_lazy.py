import pandas as pd
import xarray as xr

from monet_plots.plots.scorecard import ScorecardPlot


def test_scorecard_xarray_input():
    # 1. Create Data
    ds = xr.Dataset(
        {
            "val": (("sample"), [0.1, 0.2, 0.3, 0.4]),
            "var": (("sample"), ["O3", "PM25", "O3", "PM25"]),
            "site": (("sample"), ["A", "A", "B", "B"]),
        }
    )

    # 2. Plot
    plot = ScorecardPlot()
    plot.plot(ds, x_col="var", y_col="site", val_col="val")

    # Check that heatmap was created (it usually has a collection or image)
    assert len(plot.ax.collections) > 0
    assert "Generated ScorecardPlot" in ds.attrs["history"]


def test_scorecard_significance():
    df = pd.DataFrame(
        {
            "var": ["O3", "O3"],
            "site": ["A", "B"],
            "val": [0.1, 0.5],
            "sig": [False, True],
        }
    )

    plot = ScorecardPlot()
    plot.plot(df, x_col="var", y_col="site", val_col="val", sig_col="sig")

    # Check for significance marker (text)
    texts = [t.get_text() for t in plot.ax.texts]
    assert "*" in texts
