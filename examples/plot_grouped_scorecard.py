"""Make a grouped forecast scorecard from paired synthetic errors."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

from monet_plots.plots.scorecard import ScorecardPlot


def render(output: Path) -> None:
    """Export a scorecard and its source values."""
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(321)
    rows = []
    for region, region_effect in {"Coastal": -6, "Plains": 2, "Mountain": 7}.items():
        for variable, variable_effect in {
            "Temperature": -7,
            "Wind speed": 3,
            "Dew point": -1,
        }.items():
            for cycle, cycle_effect in {"00 UTC cycle": -3, "12 UTC cycle": 3}.items():
                for lead in (6, 12, 24, 48):
                    reference = rng.gamma(3, 1, 80)
                    effect = region_effect + variable_effect + cycle_effect + lead / 6
                    forecast = reference * rng.lognormal(
                        np.log1p(effect / 100), 0.24, 80
                    )
                    rows.append(
                        {
                            "region": region,
                            "variable": variable,
                            "cycle": cycle,
                            "lead": f"F{lead:02d}",
                            "change": 100 * (forecast.mean() / reference.mean() - 1),
                            "significant": ttest_rel(forecast, reference).pvalue < 0.05,
                            "paired_count": len(reference),
                        }
                    )
    data = pd.DataFrame(rows)
    data.loc[
        (data.region == "Mountain")
        & (data.variable == "Dew point")
        & (data.cycle == "12 UTC cycle"),
        "change",
    ] = np.nan
    plot = ScorecardPlot(style="journal", figsize=(9, 5.5), layout="constrained")
    plot.plot(
        data,
        "lead",
        "variable",
        "change",
        row_group="region",
        col_group="cycle",
        baseline="Control forecast",
        units="%",
        lower_is_better=True,
        sig_col="significant",
        title="Relative change in mean absolute error",
        vmin=-25,
        vmax=25,
    )
    plot.save(output / "grouped-scorecard.pdf")
    plot.save(output / "grouped-scorecard.png", dpi=180)
    plot.close()
    data.to_csv(output / "grouped-scorecard.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    render(parser.parse_args().output)
