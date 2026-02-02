import numpy as np
import xarray as xr
from typing import Optional, Any, List, Union
import pandas as pd
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
        x_col: str = "success_ratio",
        y_col: str = "pod",
        counts_cols: Optional[List[str]] = None,
        label_col: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Main plotting method.

        Parameters
        ----------
        data : Any
            Input data (pd.DataFrame, np.ndarray, xr.Dataset, xr.DataArray).
        x_col : str, optional
            Column name for Success Ratio (1-FAR), by default "success_ratio".
        y_col : str, optional
            Column name for POD, by default "pod".
        counts_cols : List[str], optional
            List of columns [hits, misses, fa, cn] to calculate metrics
            if x_col/y_col not present.
        label_col : str, optional
            Column to use for legend labels.
        **kwargs : Any
            Matplotlib kwargs.

        Returns
        -------
        matplotlib.axes.Axes
            The axes object.

        Examples
        --------
        >>> plot = PerformanceDiagramPlot()
        >>> plot.plot(df, counts_cols=['h', 'm', 'fa', 'cn'])
        """
        # Data Preparation (preserving laziness if xarray/dask)
        df_plot = self._prepare_data(data, x_col, y_col, counts_cols)

        # Validation
        self._validate_inputs(df_plot, x_col, y_col, counts_cols)

        # Plot Background (Isolines)
        self._draw_background()

        # Plot Data
        if label_col:
            # Both Pandas and Xarray support groupby
            for name, group in df_plot.groupby(label_col):
                # Matplotlib calls will trigger computation if dask-backed
                self.ax.plot(
                    group[x_col],
                    group[y_col],
                    marker="o",
                    label=name,
                    linestyle="none",
                    **kwargs,
                )
            self.ax.legend(loc="best")
        else:
            self.ax.plot(
                df_plot[x_col], df_plot[y_col], marker="o", linestyle="none", **kwargs
            )

        # Formatting
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_xlabel("Success Ratio (1-FAR)")
        self.ax.set_ylabel("Probability of Detection (POD)")
        self.ax.set_aspect("equal")

        return self.ax

    def _validate_inputs(
        self,
        data: Union[pd.DataFrame, xr.Dataset, xr.DataArray],
        x: str,
        y: str,
        counts: Optional[List[str]],
    ) -> None:
        """Validates input structure (handles DataFrame and Xarray).

        Parameters
        ----------
        data : Union[pd.DataFrame, xr.Dataset, xr.DataArray]
            The input data to validate.
        x : str
            The x-axis column/variable name.
        y : str
            The y-axis column/variable name.
        counts : Optional[List[str]]
            List of contingency table count names.

        Examples
        --------
        >>> plot = PerformanceDiagramPlot()
        >>> plot._validate_inputs(df, "sr", "pod", None)
        """
        required = counts if counts else [x, y]

        if isinstance(data, (xr.DataArray, xr.Dataset)):
            for col in required:
                if col not in data.coords and col not in getattr(data, "data_vars", []):
                    # For DataArray, the name might be the variable
                    if isinstance(data, xr.DataArray) and data.name == col:
                        continue
                    raise ValueError(f"Missing required variable/coordinate: {col}")
        else:
            validate_dataframe(data, required_columns=required)

    def _prepare_data(
        self,
        data: Any,
        x: str,
        y: str,
        counts: Optional[List[str]],
    ) -> Union[pd.DataFrame, xr.Dataset]:
        """Calculates metrics if counts provided, otherwise returns subset.

        Preserves Dask laziness for Xarray inputs.

        Parameters
        ----------
        data : Any
            The input data.
        x : str
            The Success Ratio column/variable name.
        y : str
            The POD column/variable name.
        counts : Optional[List[str]]
            List of columns [hits, misses, fa, cn].

        Returns
        -------
        Union[pd.DataFrame, xr.Dataset]
            The prepared data with metrics calculated.

        Examples
        --------
        >>> plot = PerformanceDiagramPlot()
        >>> ds_out = plot._prepare_data(ds, "sr", "pod", ["h", "m", "fa", "cn"])
        """
        if counts:
            hits_col, misses_col, fa_col, cn_col = counts
            sr = compute_success_ratio(data[hits_col], data[fa_col])
            pod = compute_pod(data[hits_col], data[misses_col])

            if isinstance(data, (xr.DataArray, xr.Dataset)):
                # Return Xarray with new variables
                if isinstance(data, xr.DataArray):
                    ds = data.to_dataset()
                else:
                    ds = data.copy()
                ds[x] = sr
                ds[y] = pod
                return ds
            else:
                df = to_dataframe(data).copy()
                df[x] = sr
                df[y] = pod
                return df
        return data

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

        # TDD Anchor: Test that contours are within 0-1 range.


# TDD Anchors (Unit Tests):
# 1. test_metric_calculation_from_counts: Provide hits/misses/fa, verify SR/POD output.
# 2. test_perfect_score_location: Ensure perfect forecast plots at (1,1).
# 3. test_missing_columns_error: Assert ValueError if cols missing.
# 4. test_background_drawing: Mock plt.contour, verify calls with correct grids.
