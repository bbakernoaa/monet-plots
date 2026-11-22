"""Create ensemble rank, ROC, and event performance figures with MONET Plots."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from monet_plots.plots.performance_diagram import PerformanceDiagramPlot
from monet_plots.plots.rank_histogram import RankHistogramPlot
from monet_plots.plots.roc_curve import ROCCurvePlot


def render(output: Path, style: str = "journal") -> None:
    """Render all three diagnostics from deterministic synthetic inputs."""
    output.mkdir(parents=True, exist_ok=True)
    extension = "pdf" if style == "journal" else "png"
    rng = np.random.default_rng(20261122)
    observations = rng.normal(size=5000)
    members = rng.normal(size=(len(observations), 10))
    ranks = pd.concat(
        [
            pd.DataFrame(
                {
                    "scenario": name,
                    "rank": (ensemble < observations[:, None]).sum(axis=1),
                }
            )
            for name, ensemble in {
                "Matched spread": members,
                "Too narrow": 0.45 * members,
                "Too wide": 2.2 * members,
                "Warm-biased": members + 1,
            }.items()
        ],
        ignore_index=True,
    )
    plot = RankHistogramPlot(style=style, figsize=(7.5, 5.2))
    plot.plot(ranks, n_members=10, facet_col="scenario")
    plot.save(output / f"ensemble-ranks.{extension}")
    plot.close()

    roc = pd.DataFrame(
        {
            "pofd": np.tile(np.linspace(0, 1, 6), 2),
            "pod": [0, 0.65, 0.8, 0.9, 0.95, 1, 0, 0.03, 0.1, 0.2, 0.35, 1],
            "model": ["Discriminating"] * 6 + ["Inverted scores"] * 6,
        }
    )
    plot = ROCCurvePlot(style=style)
    plot.plot(roc, label_col="model")
    plot.save(output / f"roc.{extension}")
    plot.close()

    contingency = pd.DataFrame(
        [
            ("Conservative", 44, 56, 12, 288),
            ("Balanced", 73, 27, 35, 265),
            ("Aggressive", 91, 9, 115, 185),
        ],
        columns=["model", "hits", "misses", "false_alarms", "correct_negatives"],
    )
    plot = PerformanceDiagramPlot(style=style)
    plot.plot(
        contingency,
        counts_cols=["hits", "misses", "false_alarms", "correct_negatives"],
        label_col="model",
    )
    plot.save(output / f"event-performance.{extension}")
    plot.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--style", choices=["journal", "operations"], default="journal")
    args = parser.parse_args()
    render(args.output, args.style)
