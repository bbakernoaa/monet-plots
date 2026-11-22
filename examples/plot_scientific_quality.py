"""Render the four scientific plot pilots from deterministic local fixtures."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from monet_plots.plots.reliability_diagram import ReliabilityDiagramPlot
from monet_plots.plots.spatial_bias_scatter import SpatialBiasScatterPlot
from monet_plots.plots.taylor_diagram import TaylorDiagramPlot
from monet_plots.plots.timeseries import TimeSeriesPlot


def render(output: Path, preset: str) -> None:
    """Render figures and write their synthetic input data."""
    output.mkdir(parents=True, exist_ok=True)
    baseline = preset == "baseline"
    style = "wiley" if baseline else preset
    times = pd.date_range("2024-01-01", periods=14, freq="D", tz="UTC")
    trend = 285 + 3 * np.sin(np.linspace(0, 3 * np.pi, len(times)))
    time_data = pd.DataFrame(
        {
            "time": times.repeat(3),
            "temperature": np.repeat(trend, 3) + np.tile([-1, 0, 1], len(times)),
            "units": "K",
        }
    )
    plot = TimeSeriesPlot(
        time_data,
        y="temperature",
        title="Daily temperature · synthetic stations",
        style=style,
    )
    plot.plot()

    def export(figure, name, data):
        data.to_csv(output / f"{name}.csv", index=False)
        if preset == "journal":
            figure.save(output / f"{name}.pdf")
        figure.save(output / f"{name}.png")
        figure.close()

    export(plot, "timeseries", time_data)

    observations = np.array([285, 286, 287, 288, 289, 290], dtype=float)
    station_data = pd.DataFrame(
        {
            "lat": [33, 35, 37, 34, 36, 38],
            "lon": [-104, -100, -96, -92, -88, -84],
            "observation": observations,
            "forecast": observations + [-2, -1, 0, 1, 2, 4],
        }
    )
    station_data.attrs["units"] = "K"
    plot = SpatialBiasScatterPlot(
        station_data,
        "observation",
        "forecast",
        coastlines=False,
        extent=(-108, -80, 30, 42),
        vmin=-3,
        vmax=3,
        style=style,
    )
    plot.plot()
    plot.ax.set_title("Station temperature bias")
    plot.ax.set_xticks([-105, -95, -85])
    plot.ax.set_xticklabels(["105°W", "95°W", "85°W"])
    plot.ax.set_yticks([32, 36, 40])
    plot.ax.set_yticklabels(["32°N", "36°N", "40°N"])
    plot.fig.subplots_adjust(right=0.78)
    export(plot, "station_bias", station_data)

    reliability_data = pd.DataFrame(
        {
            "prob": [0.1, 0.3, 0.5, 0.7, 0.9],
            "freq": [0.15, 0.25, np.nan, 0.63, 0.85],
            "count": [20, 20, 0, 19, 20],
        }
    )
    plot = ReliabilityDiagramPlot(style=style)
    plot.plot(reliability_data, climatology=37 / 79)
    plot.ax.set_title("Reliability of a temperature exceedance")
    export(plot, "reliability", reliability_data)

    taylor_data = pd.DataFrame(
        {
            "reference": observations,
            "offset": observations + 3,
            "high_spread": observations.mean()
            + (observations - observations.mean()) * 4,
            "inverted": observations[::-1],
        }
    )
    taylor_options = (
        {}
        if baseline
        else {
            "normalize": True,
            "show_error_table": True,
            "comparison_groups": [
                {
                    "label": "Temperature",
                    "reference": "reference",
                    "models": ["offset", "high_spread", "inverted"],
                    "unit": "K",
                }
            ],
        }
    )
    plot = TaylorDiagramPlot(
        taylor_data,
        var1="reference",
        var2=["offset", "high_spread", "inverted"],
        style=style,
        **taylor_options,
    )
    plot.plot()
    if not baseline:
        plot.error_table.to_csv(output / "taylor_statistics.csv", index=False)
    export(plot, "taylor", taylor_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("preset", choices=["baseline", "journal", "operations"])
    args = parser.parse_args()
    render(args.output, args.preset)
