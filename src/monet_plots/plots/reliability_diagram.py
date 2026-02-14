from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from ..plot_utils import to_dataframe, validate_dataframe
from ..verification_metrics import compute_reliability_curve
from .base import BasePlot


class ReliabilityDiagramPlot(BasePlot):
    """
    Reliability Diagram Plot (Attributes Diagram).

    Visualizes Observed Frequency vs Forecast Probability.

    Functional Requirements:
    1. Plot Observed Frequency (y-axis) vs Forecast Probability (x-axis).
    2. Draw "Perfect Reliability" diagonal (1:1).
    3. Draw "No Skill" line (horizontal at climatology/sample mean).
    4. Shade "Skill" areas (where Brier Skill Score > 0).
    5. Include inset histogram of forecast usage (Sharpness) if requested.

    Edge Cases:
    - Empty bins (no forecasts with that probability).
    - Climatology not provided (cannot draw skill regions correctly).
    """

    def __init__(self, fig=None, ax=None, **kwargs):
        super().__init__(fig=fig, ax=ax, **kwargs)

    def plot(
        self,
        data: Any,
        x_col: str = "prob",
        y_col: str = "freq",
        forecasts_col: Optional[str] = None,
        observations_col: Optional[str] = None,
        n_bins: int = 10,
        climatology: Optional[float] = None,
        label_col: Optional[str] = None,
        show_hist: bool = False,
        dim: Optional[Union[str, list[str]]] = None,
        **kwargs,
    ):
        """
        Main plotting method.

        Args:
            data: Input data.
            x_col (str): Forecast Probability bin center (for pre-binned).
            y_col (str): Observed Frequency in bin (for pre-binned).
            forecasts_col (str, optional): Column of raw forecast probabilities [0,1].
            observations_col (str, optional): Column of binary observations {0,1}.
            n_bins (int): Number of bins for reliability curve computation.
            climatology (Optional[float]): Sample climatology (mean(observations)).
            label_col (str, optional): Grouping column.
            show_hist (bool): Whether to show frequency of usage histogram.
            dim (str or list of str, optional): Dimension(s) to aggregate over.
            **kwargs: Matplotlib kwargs.
        """
        import xarray as xr

        if (
            isinstance(data, (xr.DataArray, xr.Dataset))
            and forecasts_col
            and observations_col
        ):
            # Native Xarray path - maintain laziness as long as possible
            ds = data
            f = ds[forecasts_col]
            o = ds[observations_col]

            if label_col:
                # If grouping by label_col, we do it in a loop for now as
                # ReliabilityDiagram is usually for small number of groups.
                # Each group calculation is still lazy.
                plot_data_list = []
                unique_labels = np.unique(ds[label_col].values)
                for label in unique_labels:
                    f_sub = f.where(ds[label_col] == label, drop=True)
                    o_sub = o.where(ds[label_col] == label, drop=True)

                    if climatology is None:
                        # Compute climatology for this group if not provided
                        c_val = float(o_sub.mean().compute())
                    else:
                        c_val = climatology

                    bc, of, ct = compute_reliability_curve(
                        f_sub, o_sub, n_bins=n_bins, dim=dim
                    )

                    # Aggregate remaining dimensions if any
                    if of.ndim > 1:
                        other_dims = set(of.dims) - {"bin"}
                        # Weighted mean for frequency
                        of = (of * ct).sum(dim=other_dims) / ct.sum(dim=other_dims)
                        ct = ct.sum(dim=other_dims)

                    plot_data_list.append(
                        pd.DataFrame(
                            {
                                x_col: bc.values,
                                y_col: of.values,
                                "count": ct.values,
                                label_col: label,
                            }
                        )
                    )
                    if climatology is None:
                        climatology = c_val  # Use the first group's climatology for reference lines if needed
                plot_data = pd.concat(plot_data_list)

            else:
                if climatology is None:
                    climatology = float(o.mean().compute())

                bc, of, ct = compute_reliability_curve(f, o, n_bins=n_bins, dim=dim)

                # Aggregate remaining dimensions if any
                if of.ndim > 1:
                    other_dims = set(of.dims) - {"bin"}
                    of = (of * ct).sum(dim=other_dims) / ct.sum(dim=other_dims)
                    ct = ct.sum(dim=other_dims)

                plot_data = pd.DataFrame(
                    {x_col: bc.values, y_col: of.values, "count": ct.values}
                )

        else:
            # Legacy/Fallback path
            df = to_dataframe(data)
            # Compute if raw data provided
            if forecasts_col and observations_col:
                if climatology is None:
                    climatology = float(df[observations_col].mean())
                bin_centers, obs_freq, bin_counts = compute_reliability_curve(
                    np.asarray(df[forecasts_col]),
                    np.asarray(df[observations_col]),
                    n_bins,
                )
                plot_data = pd.DataFrame(
                    {x_col: bin_centers, y_col: obs_freq, "count": bin_counts}
                )
            else:
                validate_dataframe(df, required_columns=[x_col, y_col])
                plot_data = df

        # Draw Reference Lines
        self.ax.plot([0, 1], [0, 1], "k--", label="Perfect Reliability")
        if climatology is not None:
            self.ax.axhline(
                climatology, color="gray", linestyle=":", label="Climatology"
            )
            self._draw_skill_regions(climatology)

        # Plot Data
        if label_col:
            for name, group in plot_data.groupby(label_col):
                # pop label from kwargs if it exists to avoid multiple values
                k = kwargs.copy()
                k.pop("label", None)
                self.ax.plot(group[x_col], group[y_col], marker="o", label=name, **k)
        else:
            k = kwargs.copy()
            label = k.pop("label", "Model")
            self.ax.plot(
                plot_data[x_col], plot_data[y_col], marker="o", label=label, **k
            )

        # Histogram Overlay (Sharpness)
        if show_hist and "count" in plot_data.columns:
            self._add_sharpness_histogram(plot_data, x_col)

        # Formatting
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_xlabel("Forecast Probability")
        self.ax.set_ylabel("Observed Relative Frequency")
        self.ax.set_aspect("equal")
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()

        return self.ax

    def _draw_skill_regions(self, clim):
        """Shades areas where BSS > 0."""
        x = np.linspace(0, 1, 100)
        y_no_skill = np.full_like(x, clim)
        y_perfect = x

        # Shade skill region (above no-skill towards perfect)
        self.ax.fill_between(
            x, y_no_skill, y_perfect, alpha=0.1, color="green", label="Skill Region"
        )

    def _add_sharpness_histogram(self, data, x_col):
        """Adds a small inset axes for sharpness histogram."""
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        inset_ax = inset_axes(self.ax, width=1.5, height=1.2, loc="upper right")
        inset_ax.bar(data[x_col], data["count"], alpha=0.5, color="blue", width=0.08)
        inset_ax.set_title("Sharpness")
        inset_ax.set_xlabel(x_col)
        inset_ax.set_ylabel("Count")
