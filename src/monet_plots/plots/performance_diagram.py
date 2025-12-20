import numpy as np
from typing import Optional, List, Any
from .base import BasePlot
from ..plot_utils import validate_dataframe, to_dataframe
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

    def __init__(self, fig=None, ax=None, **kwargs):
        super().__init__(fig=fig, ax=ax, **kwargs)

    def plot(
        self,
        data: Any,
        x_col: str = 'success_ratio',
        y_col: str = 'pod',
        counts_cols: Optional[List[str]] = None,
        label_col: Optional[str] = None,
        **kwargs,
    ):
        """
        Main plotting method.

        Args:
            data (pd.DataFrame, np.ndarray, xr.Dataset, xr.DataArray): Input data.
            x_col (str): Column name for Success Ratio (1-FAR).
            y_col (str): Column name for POD.
            counts_cols (list, optional): List of columns [hits, misses, fa, cn]
                                        to calculate metrics if x_col/y_col not present.
            label_col (str, optional): Column to use for legend labels.
            **kwargs: Matplotlib kwargs.
        """
        df = to_dataframe(data)
        # TDD Anchor: Test validation raises error on missing cols
        self._validate_inputs(df, x_col, y_col, counts_cols)

        # Data Preparation
        df_plot = self._prepare_data(df, x_col, y_col, counts_cols)

        # Plot Background (Isolines)
        self._draw_background()

        # Plot Data
        # TDD Anchor: Verify scatter points match input data coordinates
        if label_col:
            for name, group in df_plot.groupby(label_col):
                self.ax.plot(group[x_col], group[y_col], marker='o',
                             label=name, linestyle='none', **kwargs)
            self.ax.legend(loc='best')
        else:
            self.ax.plot(df_plot[x_col], df_plot[y_col],
                         marker='o', linestyle='none', **kwargs)

        # Formatting
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_xlabel("Success Ratio (1-FAR)")
        self.ax.set_ylabel("Probability of Detection (POD)")
        self.ax.set_aspect('equal')

    def _validate_inputs(self, data, x, y, counts):
        """Validates input dataframe structure."""
        if counts:
            validate_dataframe(data, required_columns=counts)
        else:
            validate_dataframe(data, required_columns=[x, y])

    def _prepare_data(self, data, x, y, counts):
        """
        Calculates metrics if counts provided, otherwise returns subset.
        TDD Anchor: Test calculation logic: SR = hits/(hits+fa), POD = hits/(hits+miss).
        """
        df = data.copy()
        if counts:
            hits_col, misses_col, fa_col, cn_col = counts
            df[x] = compute_success_ratio(df[hits_col], df[fa_col])
            df[y] = compute_pod(df[hits_col], df[misses_col])
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
        xx, yy = np.meshgrid(np.linspace(0.01, 0.99, 50),
                             np.linspace(0.01, 0.99, 50))
        csi = (xx * yy) / (xx + yy - xx * yy)
        bias = yy / xx

        # CSI contours (dashed, lightgray)
        cs_csi = self.ax.contour(xx, yy, csi, levels=np.arange(0.1, 0.95, 0.1),
                                 colors='lightgray', linestyles='--', alpha=0.6)
        self.ax.clabel(cs_csi, inline=True, fontsize=8, fmt='%.1f')

        # Bias contours (dotted, darkgray)
        cs_bias = self.ax.contour(xx, yy, bias, levels=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
                                  colors='darkgray', linestyles=':', alpha=0.6)
        self.ax.clabel(cs_bias, inline=True, fontsize=8, fmt='%.1f')

        # Perfect forecast line
        self.ax.plot([0.01, 0.99], [0.01, 0.99],
                     'k-', linewidth=1.5, alpha=0.8)

        # TDD Anchor: Test that contours are within 0-1 range.

# TDD Anchors (Unit Tests):
# 1. test_metric_calculation_from_counts: Provide hits/misses/fa, verify SR/POD output.
# 2. test_perfect_score_location: Ensure perfect forecast plots at (1,1).
# 3. test_missing_columns_error: Assert ValueError if cols missing.
# 4. test_background_drawing: Mock plt.contour, verify calls with correct grids.
