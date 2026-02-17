import numpy as np
import pandas as pd
from typing import Optional, Any, List
from .base import BasePlot
from ..plot_utils import to_dataframe
from ..verification_metrics import compute_pod, compute_success_ratio


class PerformanceDiagramPlot(BasePlot):
    """
    Performance Diagram Plot (Roebber).

    Visualizes the relationship between Probability of Detection (POD),
    Success Ratio (SR),
    Critical Success Index (CSI), and Bias.

    Functional Requirements:
    1. Plot POD (y-axis) vs Success Ratio (x-axis).
    2. Draw background isolines for CSI and Bias.
    3. Support input as pre-calculated metrics or contingency table counts.
    4. Handle multiple models/configurations via grouping.

    Edge Cases:
    - SR or POD being 0 or 1 (division by zero in bias/CSI calculations).
    - Empty DataFrame.
    - Missing required columns.
    """

    def __init__(
        self,
        data: Any,
        x_col: str = "success_ratio",
        y_col: str = "pod",
        counts_cols: Optional[List[str]] = None,
        obs_col: Optional[str] = None,
        mod_col: Optional[str] = None,
        thresholds: Optional[List[float]] = None,
        label_col: Optional[str] = None,
        fig=None,
        ax=None,
        **kwargs,
    ):
        super().__init__(fig=fig, ax=ax, **kwargs)
        self.data = data
        self.x_col = x_col
        self.y_col = y_col
        self.counts_cols = counts_cols
        self.obs_col = obs_col
        self.mod_col = mod_col
        self.thresholds = thresholds
        self.label_col = label_col

    def plot(self, **kwargs):
        """
        Main plotting method.

        Args:
            **kwargs: Matplotlib kwargs.
        """
        # Data Preparation
        df_plot = self._prepare_data()

        # Plot Background (Isolines)
        self._draw_background()

        # Plot Data
        if self.label_col:
            for name, group in df_plot.groupby(self.label_col):
                self.ax.plot(
                    group[self.x_col],
                    group[self.y_col],
                    marker="o",
                    label=name,
                    linestyle="none",
                    **kwargs,
                )
            self.ax.legend(loc="best")
        else:
            self.ax.plot(
                df_plot[self.x_col],
                df_plot[self.y_col],
                marker="o",
                linestyle="none",
                **kwargs,
            )

        # Formatting
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_xlabel("Success Ratio (1-FAR)")
        self.ax.set_ylabel("Probability of Detection (POD)")
        self.ax.set_aspect("equal")

    def _prepare_data(self):
        """
        Prepares data for plotting, calculating metrics from raw data or counts if necessary.
        """
        from ..verification_metrics import (
            compute_categorical_metrics,
            compute_contingency_table,
        )

        if self.obs_col and self.mod_col and self.thresholds:
            # Plotting from raw data at various thresholds
            obs = self.data[self.obs_col]
            mod = self.data[self.mod_col]

            rows = []
            for t in self.thresholds:
                ct = compute_contingency_table(obs, mod, t)
                # If dask-backed, compute now for plotting
                if hasattr(ct["hits"], "compute"):
                    import dask

                    ct = dask.compute(ct)[0]

                metrics = compute_categorical_metrics(**ct)
                metrics["threshold"] = t
                rows.append(metrics)

            df_plot = pd.DataFrame(rows)
            # Threshold as label if label_col not provided
            if self.label_col is None:
                self.label_col = "threshold"
            return df_plot

        # Ensure we have a DataFrame for simpler plotting logic if not already
        df = to_dataframe(self.data)

        if self.counts_cols:
            hits_col, misses_col, fa_col, cn_col = self.counts_cols
            df[self.x_col] = compute_success_ratio(df[hits_col], df[fa_col])
            df[self.y_col] = compute_pod(df[hits_col], df[misses_col])

        return df

    def _draw_background(self):
        """
        Draws CSI and Bias isolines.

        Pseudocode:
        1. Create meshgrid for x (SR) and y (POD) from 0.01 to 1.
        2. Calculate CSI = 1 / (1/SR + 1/POD - 1).
        3. Calculate Bias = POD / SR.
        4. Contour plot CSI (dashed).
        5. Contour plot Bias (dotted).
        6. Label contours.
        """
        # Avoid division by zero at boundaries
        xx, yy = np.meshgrid(np.linspace(0.01, 0.99, 50), np.linspace(0.01, 0.99, 50))
        csi = (xx * yy) / (xx + yy - xx * yy)
        bias = yy / xx

        # CSI contours (dashed, lightgray)
        cs_csi = self.ax.contour(
            xx,
            yy,
            csi,
            levels=np.arange(0.1, 0.95, 0.1),
            colors="lightgray",
            linestyles="--",
            alpha=0.6,
        )
        self.ax.clabel(cs_csi, inline=True, fontsize=8, fmt="%.1f")

        # Bias contours (dotted, darkgray)
        cs_bias = self.ax.contour(
            xx,
            yy,
            bias,
            levels=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
            colors="darkgray",
            linestyles=":",
            alpha=0.6,
        )
        self.ax.clabel(cs_bias, inline=True, fontsize=8, fmt="%.1f")

        # Perfect forecast line
        self.ax.plot([0.01, 0.99], [0.01, 0.99], "k-", linewidth=1.5, alpha=0.8)

    def hvplot(self, **kwargs):
        """Generate an interactive performance diagram using hvPlot."""
        import holoviews as hv
        import hvplot.pandas  # noqa: F401

        df_plot = self._prepare_data()

        plot_kwargs = {
            "x": self.x_col,
            "y": self.y_col,
            "kind": "scatter",
            "xlabel": "Success Ratio (1-FAR)",
            "ylabel": "Probability of Detection (POD)",
            "xlim": (0, 1),
            "ylim": (0, 1),
        }
        if self.label_col:
            plot_kwargs["by"] = self.label_col

        plot_kwargs.update(kwargs)

        # Background isolines (simplified for HoloViews)
        xx, yy = np.meshgrid(np.linspace(0.01, 0.99, 100), np.linspace(0.01, 0.99, 100))
        csi = (xx * yy) / (xx + yy - xx * yy)
        # bias = yy / xx

        csi_contours = hv.operation.contours(
            hv.Image(csi, bounds=(0, 0, 1, 1)),
            levels=np.arange(0.1, 0.95, 0.1).tolist(),
        ).opts(alpha=0.3, cmap=["gray"], line_dash="dashed")
        perfect_line = hv.Curve([(0, 0), (1, 1)]).opts(color="black", alpha=0.5)

        return (csi_contours * perfect_line * df_plot.hvplot(**plot_kwargs)).opts(
            title="Performance Diagram"
        )


# 4. test_background_drawing: Mock plt.contour, verify calls with correct grids.
