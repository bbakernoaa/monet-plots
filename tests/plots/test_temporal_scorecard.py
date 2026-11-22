import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from monet_plots import TemporalScorecardPlot


def test_grouped_temporal_scorecard_keeps_chronology_and_empty_periods(tmp_path):
    data = pd.DataFrame(
        {
            "time": pd.to_datetime(
                ["2025-01-03", "2025-01-01", "2025-01-01"], utc=True
            ),
            "region": ["West", "East", "West"],
            "variable": ["Wind", "Wind", "Wind"],
            "score": [3.0, -2.0, 1.0],
        }
    )
    plot = TemporalScorecardPlot()
    try:
        ax = plot.plot(
            data, "time", "variable", "score", row_group="region", frequency="D"
        )
        matrix = ax.collections[0].get_array()
        assert matrix.shape == (2, 3)
        assert matrix.mask[:, 1].all()
        np.testing.assert_allclose(matrix[1, [0, 2]], [1, 3])
        assert "UTC" in ax.get_xlabel()
        assert "02 Jan" in [label.get_text() for label in ax.get_xticklabels()]
        assert [label.get_text() for label in ax.get_yticklabels()] == [
            "East  /  Wind",
            "West  /  Wind",
        ]
        plot.save(tmp_path / "temporal-scorecard.pdf")
    finally:
        plt.close(plot.fig)


def test_temporal_scorecard_rejects_off_grid_timestamps():
    data = pd.DataFrame(
        {
            "time": ["2025-01-01", "2025-01-02 06:00"],
            "region": ["A", "A"],
            "score": [1, 2],
        }
    )
    plot = TemporalScorecardPlot()
    try:
        with pytest.raises(ValueError, match="frequency"):
            plot.plot(data, "time", "region", "score", frequency="D")
    finally:
        plt.close(plot.fig)
