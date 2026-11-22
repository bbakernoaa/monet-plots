"""Render region and grouped-variable scorecards over valid time."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from monet_plots import TemporalScorecardPlot


def render(output: Path) -> None:
    """Write both examples and the synthetic values behind them."""
    output.mkdir(parents=True, exist_ok=True)
    days = pd.date_range("2025-07-01", periods=12, freq="D", tz="UTC")
    step = np.arange(len(days))
    region_scores = {
        "Coastal": 4 * np.sin(step / 2),
        "Plains": -8 + 1.4 * step,
        "Mountain": np.where(step >= 7, 12, -3),
        "Great Lakes": -2 + 5 * np.cos(step / 2),
    }
    regions = pd.concat(
        [
            pd.DataFrame({"valid_time": days, "region": name, "change": values})
            for name, values in region_scores.items()
        ],
        ignore_index=True,
    )
    regions = regions.loc[regions.valid_time != days[5]]

    times = pd.date_range("2025-07-01", periods=16, freq="6h", tz="UTC")
    phase = np.arange(len(times)) / 3
    variables = pd.concat(
        [
            pd.DataFrame(
                {
                    "valid_time": times,
                    "region": region,
                    "variable": variable,
                    "change": offset + amplitude * np.sin(phase + shift),
                }
            )
            for region, shift in [("Coastal", 0), ("Mountain", 2.2)]
            for variable, offset, amplitude in [
                ("Temperature", -3, 10),
                ("Wind speed", 4, 6),
                ("Dew point", 0, 5),
            ]
        ],
        ignore_index=True,
    )
    variables.loc[
        variables.variable.eq("Wind speed")
        & variables.region.eq("Mountain")
        & variables.valid_time.isin(times[4:7]),
        "change",
    ] = np.nan

    for data, y_col, group, freq, name, title in [
        (
            regions,
            "region",
            None,
            "D",
            "region-by-time",
            "Daily forecast error by region",
        ),
        (
            variables,
            "variable",
            "region",
            "6h",
            "grouped-variables-by-time",
            "Six-hour forecast error by variable and region",
        ),
    ]:
        plot = TemporalScorecardPlot(figsize=(10, 4.4))
        plot.plot(
            data,
            "valid_time",
            y_col,
            "change",
            row_group=group,
            frequency=freq,
            baseline="Control forecast",
            lower_is_better=True,
            units="%",
            vmin=-15,
            vmax=15,
            title=title,
        )
        plot.save(output / f"{name}.pdf")
        plot.save(output / f"{name}.png", dpi=180)
        plot.close()
        data.to_csv(output / f"{name}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    render(parser.parse_args().output)
