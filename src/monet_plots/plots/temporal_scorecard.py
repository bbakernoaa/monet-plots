"""Scorecards showing how verification metrics change across valid times."""

from typing import Any

import numpy as np
import pandas as pd

from ..plot_utils import to_dataframe, validate_dataframe
from .scorecard import ScorecardPlot


class TemporalScorecardPlot(ScorecardPlot):
    """Plot regions or grouped variables against regular verification periods.

    Each cell is a precomputed score for one row and time. Use the categorical
    ScorecardPlot for lead times, forecast cycle groups, or other column categories.
    """

    _grouped_by_default = True

    def __init__(self, *, style: str | None = "journal", **kwargs):
        super().__init__(style=style, **kwargs)

    def plot(
        self,
        data: Any,
        time_col: str,
        y_col: str,
        val_col: str,
        *,
        row_group: str | None = None,
        frequency: str | None = None,
        **kwargs,
    ):
        """Render a time scorecard and return its axes.

        Set `row_group` to group variables under regions, or leave it unset
        to give each region or variable one row. Set `frequency` to a pandas
        date frequency such as "D" or "6h" to insert empty periods. If
        omitted, the supplied valid times must be regular. Time zone metadata
        is retained if supplied. Scorecard options such as `baseline`, `units`,
        `sig_col`, and explicit color scale limits are passed through.
        """
        frame = to_dataframe(data).copy()
        required = [time_col, y_col, val_col] + ([row_group] if row_group else [])
        validate_dataframe(frame, required_columns=required)
        if pd.api.types.is_numeric_dtype(frame[time_col]):
            raise ValueError("Use actual datetimes for valid time.")
        try:
            times = pd.DatetimeIndex(pd.to_datetime(frame[time_col], format="mixed"))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Use valid timestamps with a consistent time zone."
            ) from exc
        if times.hasnans:
            raise ValueError("Each scorecard row needs a valid timestamp.")
        frame[time_col] = times
        periods = times.unique().sort_values()

        if frequency is not None:
            calendar = pd.date_range(periods[0], periods[-1], freq=frequency)
            if not periods.isin(calendar).all():
                raise ValueError(
                    "Some timestamps do not align with the requested frequency."
                )
            missing = calendar.difference(periods)
            if len(missing):
                # A single row key per absent timestamp introduces the entire empty column.
                keys = [y_col] + ([row_group] if row_group else [])
                fillers = pd.DataFrame(
                    {key: frame[key].iloc[0] for key in keys}, index=missing
                )
                fillers[time_col] = missing
                frame = pd.concat([frame, fillers], ignore_index=True)
            periods = calendar
        elif len(periods) > 2 and pd.infer_freq(periods) is None:
            raise ValueError(
                "Irregular timestamps require a frequency to show empty periods."
            )

        frame = frame.sort_values(time_col, kind="stable")
        frame[time_col] = periods.get_indexer(frame[time_col])
        kwargs.setdefault("title", "Performance by valid time")
        kwargs.setdefault("annot", len(periods) <= 16)
        kwargs["cbar_kws"] = {"pad": 0.19, **kwargs.get("cbar_kws", {})}
        super().plot(frame, time_col, y_col, val_col, row_group=row_group, **kwargs)
        subdaily = not periods.equals(periods.normalize())
        max_labels = max(
            2, min(8 if subdaily else 12, int(self.fig.get_figwidth() * 1.3))
        )
        stride = (len(periods) + max_labels - 1) // max_labels
        indices = np.arange(0, len(periods), stride)
        pattern = "%d %b\n%H:%M" if subdaily else "%d %b"
        if periods[0].year != periods[-1].year:
            pattern = "%d %b %y\n%H:%M" if subdaily else "%d %b %y"
        self.ax.set_xticks(indices + 0.5, periods[indices].strftime(pattern))
        self.ax.xaxis.tick_bottom()
        context = [str(times.tz)] if times.tz is not None else []
        if periods[0].year == periods[-1].year:
            context.append(str(periods[0].year))
        context_label = f" ({', '.join(context)})" if context else ""
        self.ax.set_xlabel(f"Valid time{context_label} · gray cells: missing")
        return self.ax
