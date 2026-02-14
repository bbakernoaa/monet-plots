import dask.array as da
import numpy as np
import xarray as xr

from monet_plots.plots.roc_curve import ROCCurvePlot


def test_roc_curve_lazy_parity():
    # 1. Create Data (ROC points)
    pofd = np.array([0, 0.2, 0.5, 0.8, 1.0])
    pod = np.array([0, 0.4, 0.7, 0.9, 1.0])

    ds_eager = xr.Dataset({"pofd": (("point"), pofd), "pod": (("point"), pod)})

    ds_lazy = xr.Dataset(
        {
            "pofd": (("point"), da.from_array(pofd, chunks=2)),
            "pod": (("point"), da.from_array(pod, chunks=2)),
        }
    )

    # 2. Plot Eager
    plot_eager = ROCCurvePlot()
    plot_eager.plot(ds_eager, x_col="pofd", y_col="pod")

    # 3. Plot Lazy
    plot_lazy = ROCCurvePlot()
    plot_lazy.plot(ds_lazy, x_col="pofd", y_col="pod")

    # 4. Compare Plot Data
    # ax.lines[0] is diagonal, [1] is the ROC curve
    x_eager, y_eager = plot_eager.ax.lines[1].get_data()
    x_lazy, y_lazy = plot_lazy.ax.lines[1].get_data()

    np.testing.assert_allclose(x_eager, x_lazy)
    np.testing.assert_allclose(y_eager, y_lazy)

    # Check labels (including AUC)
    label_eager = plot_eager.ax.lines[1].get_label()
    label_lazy = plot_lazy.ax.lines[1].get_label()
    assert label_eager == label_lazy
    assert "AUC" in label_lazy

    assert "Generated ROCCurvePlot" in ds_lazy.attrs["history"]


def test_roc_curve_grouping_lazy():
    pofd = np.array([0, 0.5, 1.0, 0, 0.4, 1.0])
    pod = np.array([0, 0.6, 1.0, 0, 0.5, 1.0])
    labels = np.array(["A", "A", "A", "B", "B", "B"])

    ds_lazy = xr.Dataset(
        {
            "pofd": (("point"), da.from_array(pofd, chunks=3)),
            "pod": (("point"), da.from_array(pod, chunks=3)),
            "model": (("point"), labels),
        }
    )

    plot = ROCCurvePlot()
    plot.plot(ds_lazy, x_col="pofd", y_col="pod", label_col="model")

    # Diagonal + 2 curves = 3 lines
    assert len(plot.ax.lines) == 3
    labels = [line.get_label() for line in plot.ax.lines]
    assert any("A" in label_str for label_str in labels)
    assert any("B" in label_str for label_str in labels)
