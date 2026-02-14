from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import pandas as pd

from ..plot_utils import to_dataframe, validate_dataframe
from ..verification_metrics import compute_brier_score_components
from .base import BasePlot

if TYPE_CHECKING:
    import matplotlib.axes


class BrierScoreDecompositionPlot(BasePlot):
    """
    Brier Score Decomposition Plot.

    Visualizes the components of the Brier Score: Reliability,
    Resolution, and Uncertainty.
    BS = Reliability - Resolution + Uncertainty
    """

    def __init__(self, fig=None, ax=None, **kwargs):
        super().__init__(fig=fig, ax=ax, **kwargs)

    def plot(
        self,
        data: Any,
        reliability_col: str = "reliability",
        resolution_col: str = "resolution",
        uncertainty_col: str = "uncertainty",
        forecasts_col: Optional[str] = None,
        observations_col: Optional[str] = None,
        n_bins: int = 10,
        label_col: Optional[str] = None,
        dim: Optional[Union[str, list[str]]] = None,
        **kwargs: Any,
    ) -> matplotlib.axes.Axes:
        """
        Main plotting method.

        Parameters
        ----------
        data : Any
            Input data. Can be a pandas DataFrame, xarray DataArray,
            xarray Dataset, or numpy ndarray.
        reliability_col : str, optional
            Pre-computed reliability column name, by default "reliability".
        resolution_col : str, optional
            Pre-computed resolution column name, by default "resolution".
        uncertainty_col : str, optional
            Pre-computed uncertainty column name, by default "uncertainty".
        forecasts_col : str, optional
            Column of raw forecast probabilities [0, 1].
        observations_col : str, optional
            Column of binary observations {0, 1}.
        n_bins : int, optional
            Number of bins for decomposition if raw data is provided, by default 10.
        label_col : str, optional
            Grouping column for multiple decompositions.
        dim : Union[str, list[str]], optional
            Dimension(s) to aggregate over when using Xarray/Dask raw data.
        **kwargs : Any
            Additional keyword arguments passed to ax.bar or suptitle.

        Returns
        -------
        matplotlib.axes.Axes
            The axes object with the Brier Score decomposition plot.
        """
        import xarray as xr

        # Track provenance if Xarray
        if isinstance(data, (xr.DataArray, xr.Dataset)):
            history = data.attrs.get("history", "")
            data.attrs["history"] = f"Generated BrierScoreDecompositionPlot; {history}"

        title = kwargs.pop("title", "Brier Score Decomposition")

        if (
            isinstance(data, (xr.DataArray, xr.Dataset))
            and forecasts_col
            and observations_col
        ):
            # Native Xarray path
            ds = data
            f = ds[forecasts_col]
            o = ds[observations_col]

            components_list = []
            if label_col:
                labels_concrete = ds[label_col].values
                unique_labels = np.unique(labels_concrete)
                for label in unique_labels:
                    # Select using mask for boolean dask indexing safety
                    mask = labels_concrete == label
                    f_sub = f.isel({f.dims[0]: mask})
                    o_sub = o.isel({o.dims[0]: mask})

                    comps = compute_brier_score_components(
                        f_sub, o_sub, n_bins=n_bins, dim=dim
                    )
                    # Comps might be lazy, need to compute for the bar plot
                    row = {
                        k: float(v.compute()) if hasattr(v, "compute") else float(v)
                        for k, v in comps.items()
                    }
                    row["model"] = str(label)
                    components_list.append(row)
            else:
                comps = compute_brier_score_components(f, o, n_bins=n_bins, dim=dim)
                row = {
                    k: float(v.compute()) if hasattr(v, "compute") else float(v)
                    for k, v in comps.items()
                }
                row["model"] = "Model"
                components_list.append(row)

            df_plot = pd.DataFrame(components_list)
            plot_label_col = "model"
        else:
            # Fallback path
            df = to_dataframe(data)
            # Compute components if raw data provided
            if forecasts_col and observations_col:
                components_list = []
                if label_col:
                    for name, group in df.groupby(label_col):
                        comps = compute_brier_score_components(
                            np.asarray(group[forecasts_col]),
                            np.asarray(group[observations_col]),
                            n_bins,
                        )
                        row = pd.Series(comps)
                        row["model"] = str(name)
                        components_list.append(row)
                else:
                    comps = compute_brier_score_components(
                        np.asarray(df[forecasts_col]),
                        np.asarray(df[observations_col]),
                        n_bins,
                    )
                    row = pd.Series(comps)
                    row["model"] = "Model"
                    components_list.append(row)

                df_plot = pd.DataFrame(components_list)
                plot_label_col = "model"
            else:
                required_cols = [reliability_col, resolution_col, uncertainty_col]
                validate_dataframe(df, required_columns=required_cols)
                df_plot = df
                plot_label_col = label_col

        # Prepare for plotting: make resolution negative for visualization
        df_plot = df_plot.copy()
        df_plot["resolution_plot"] = -df_plot[resolution_col]

        # Grouped bar plot
        if plot_label_col:
            labels = df_plot[plot_label_col].astype(str)
        else:
            labels = df_plot.index.astype(str)

        x = np.arange(len(labels))
        width = 0.25

        self.ax.bar(
            x - width,
            df_plot[reliability_col],
            width,
            label="Reliability",
            color="red",
            alpha=0.8,
            **kwargs,
        )
        self.ax.bar(
            x,
            df_plot["resolution_plot"],
            width,
            label="Resolution (-)",
            color="green",
            alpha=0.8,
            **kwargs,
        )
        self.ax.bar(
            x + width,
            df_plot[uncertainty_col],
            width,
            label="Uncertainty",
            color="blue",
            alpha=0.8,
            **kwargs,
        )

        # Total Brier Score as line on top if available
        if "brier_score" in df_plot.columns:
            self.ax.plot(
                x,
                df_plot["brier_score"],
                "ko-",
                linewidth=2,
                markersize=6,
                label="Brier Score",
            )

        self.ax.set_xticks(x)
        self.ax.set_xticklabels(labels, rotation=45, ha="right")
        self.ax.legend(loc="best")
        self.ax.set_ylabel("Brier Score Components")
        self.ax.set_title(title)
        self.ax.grid(True, alpha=0.3)
