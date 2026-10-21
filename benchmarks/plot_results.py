"""Plot recorded benchmark timings and process-memory measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_report(report, output):
    results = report["results"]
    positions = np.arange(len(results))
    labels = [item["case"]["name"].replace("_", " ") for item in results]
    construction = np.array([item["median"]["construct_s"] for item in results])
    drawing = np.array([item["median"]["draw_s"] for item in results])
    memory = np.array([item["median"]["peak_rss_increase_mib"] for item in results])
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9}):
        fig, (left, right) = plt.subplots(
            1,
            2,
            figsize=(11, max(4, len(results) * 0.4)),
            sharey=True,
            layout="constrained",
        )
        left.barh(
            positions,
            construction,
            label="Scale selection / artist construction",
            color="#0072B2",
        )
        left.barh(
            positions, drawing, left=construction, label="Canvas draw", color="#E69F00"
        )
        right.barh(positions, memory, color="#009E73")
        for index, result in enumerate(results):
            for run in result["runs"]:
                left.plot(run["render_s"], index, "|", color="black", markersize=8)
                right.plot(
                    run["peak_rss_increase_mib"],
                    index,
                    "|",
                    color="black",
                    markersize=8,
                )
        left.set_yticks(positions, labels)
        left.invert_yaxis()
        left.set_xlabel("Render time (seconds)")
        right.set_xlabel("Peak RSS increase above warmed baseline (MiB)")
        left.legend(loc="lower right", fontsize=7)
        for ax in (left, right):
            ax.grid(axis="x", alpha=0.2)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle("Spatial plotting: isolated-process benchmark")
        fig.supxlabel(
            f"Medians of {report['repeat']} runs; black ticks show individual measurements. Synthetic float32 fields; no map features. PNG export is excluded.",
            fontsize=8,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmarks.png"))
    args = parser.parse_args()
    plot_report(json.loads(args.report.read_text()), args.output)


if __name__ == "__main__":
    main()
