import numpy as np
import pandas as pd
from monet_plots.tools import calc_24hr_ave, calc_8hr_rolling_max


def test_regulatory_calcs():
    """Test regulatory calculation helpers in tools.py."""
    dates = pd.date_range("2023-01-01", periods=48, freq="h")
    data = np.ones(48)

    df = pd.DataFrame({"time_local": dates, "obs": data, "siteid": ["site1"] * 48})

    # Test 24hr average
    res_24 = calc_24hr_ave(df, col="obs")
    assert not res_24.empty
    assert res_24["obs_reg"].iloc[0] == 1.0

    # Test 8hr rolling max
    res_8 = calc_8hr_rolling_max(df, col="obs")
    assert not res_8.empty
    assert res_8["obs_reg"].iloc[0] == 1.0
