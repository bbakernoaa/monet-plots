import numpy as np
import pandas as pd
import xarray as xr
import holoviews as hv
from monet_plots import (
    ScatterPlot,
    TimeSeriesPlot,
    SpatialImshowPlot,
    SoccerPlot,
    PerformanceDiagramPlot,
    ROCCurvePlot,
    ReliabilityDiagramPlot,
    RankHistogramPlot,
    BrierScoreDecompositionPlot,
    ScorecardPlot,
    RelativeEconomicValuePlot,
    ConditionalBiasPlot,
    KDEPlot,
    RidgelinePlot,
    TaylorDiagramPlot,
    CurtainPlot,
    FingerprintPlot,
    TrajectoryPlot,
    DiurnalErrorPlot,
    WindBarbsPlot,
    WindQuiverPlot,
    Windrose,
    Meteogram,
    UpperAir,
    ProfilePlot,
    FacetGridPlot,
    BivariatePolarPlot,
    StickPlot,
    VerticalBoxPlot,
)


def test_scatter_hvplot():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6], "c": [7, 8, 9]})
    plot = ScatterPlot(df, x="x", y="y", c="c")
    hv_obj = plot.hvplot()
    assert isinstance(hv_obj, (hv.Element, hv.Layout, hv.Overlay, hv.NdLayout))


def test_timeseries_hvplot():
    df = pd.DataFrame(
        {"time": pd.date_range("2023-01-01", periods=3), "obs": [1, 2, 3]}
    )
    plot = TimeSeriesPlot(df, x="time", y="obs")
    hv_obj = plot.hvplot()
    assert isinstance(hv_obj, (hv.Element, hv.Layout, hv.Overlay, hv.NdLayout))


def test_spatial_imshow_hvplot():
    lon = np.linspace(-120, -70, 10)
    lat = np.linspace(20, 50, 10)
    data = np.random.rand(10, 10)
    da = xr.DataArray(data, coords=[lat, lon], dims=["lat", "lon"], name="test")
    plot = SpatialImshowPlot(da)
    hv_obj = plot.hvplot()
    assert isinstance(hv_obj, (hv.Element, hv.Layout, hv.Overlay, hv.NdLayout))


def test_soccer_hvplot():
    df = pd.DataFrame({"obs": [1, 2, 3], "mod": [1.1, 1.9, 3.2]})
    plot = SoccerPlot(df, obs_col="obs", mod_col="mod")
    hv_obj = plot.hvplot()
    assert isinstance(hv_obj, (hv.Element, hv.Layout, hv.Overlay, hv.NdLayout))


def test_performance_diagram_hvplot():
    df = pd.DataFrame({"success_ratio": [0.8, 0.9], "pod": [0.7, 0.85]})
    plot = PerformanceDiagramPlot(df)
    hv_obj = plot.hvplot()
    assert isinstance(hv_obj, (hv.Element, hv.Layout, hv.Overlay, hv.NdLayout))


def test_roc_curve_hvplot():
    df = pd.DataFrame({"pofd": [0.1, 0.2, 0.5], "pod": [0.3, 0.6, 0.8]})
    plot = ROCCurvePlot(df)
    hv_obj = plot.hvplot()
    assert isinstance(hv_obj, (hv.Element, hv.Layout, hv.Overlay, hv.NdLayout))


def test_reliability_diagram_hvplot():
    df = pd.DataFrame({"prob": [0.1, 0.5, 0.9], "freq": [0.12, 0.48, 0.95]})
    plot = ReliabilityDiagramPlot(df)
    hv_obj = plot.hvplot()
    assert isinstance(hv_obj, (hv.Element, hv.Layout, hv.Overlay, hv.NdLayout))


def test_rank_histogram_hvplot():
    df = pd.DataFrame({"rank": [1, 2, 3, 1, 2, 0]})
    plot = RankHistogramPlot(df)
    hv_obj = plot.hvplot()
    assert isinstance(hv_obj, (hv.Element, hv.Layout, hv.Overlay, hv.NdLayout))


def test_brier_decomposition_hvplot():
    df = pd.DataFrame(
        {"reliability": [0.1], "resolution": [0.2], "uncertainty": [0.15]}
    )
    plot = BrierScoreDecompositionPlot(df)
    hv_obj = plot.hvplot()
    assert isinstance(hv_obj, (hv.Element, hv.Layout, hv.Overlay, hv.NdLayout))


def test_scorecard_hvplot():
    df = pd.DataFrame({"x": ["A", "B"], "y": ["C", "D"], "val": [1, 2]})
    plot = ScorecardPlot(df, x_col="x", y_col="y", val_col="val")
    hv_obj = plot.hvplot()
    assert isinstance(hv_obj, (hv.Element, hv.Layout, hv.Overlay, hv.NdLayout))


def test_rev_hvplot():
    df = pd.DataFrame({"hits": [10], "misses": [5], "fa": [2], "cn": [20]})
    plot = RelativeEconomicValuePlot(df)
    hv_obj = plot.hvplot()
    assert isinstance(hv_obj, (hv.Element, hv.Layout, hv.Overlay, hv.NdLayout))


def test_conditional_bias_hvplot():
    df = pd.DataFrame(
        {"obs": [1, 2, 3, 4, 5, 6], "fcst": [1.1, 1.9, 3.2, 3.8, 5.1, 6.2]}
    )
    plot = ConditionalBiasPlot(df, obs_col="obs", fcst_col="fcst")
    hv_obj = plot.hvplot()
    assert isinstance(hv_obj, (hv.Element, hv.Layout, hv.Overlay, hv.NdLayout))


def test_kde_hvplot():
    df = pd.DataFrame({"x": [1, 2, 3, 2, 1], "y": [4, 5, 6, 5, 4]})
    plot = KDEPlot(df, x="x", y="y")
    hv_obj = plot.hvplot()
    assert isinstance(hv_obj, (hv.Element, hv.Layout, hv.Overlay, hv.NdLayout))


def test_ridgeline_hvplot():
    df = pd.DataFrame({"val": [1, 2, 3, 4], "group": ["A", "A", "B", "B"]})
    plot = RidgelinePlot(df, group_dim="group", x="val")
    hv_obj = plot.hvplot()
    assert isinstance(hv_obj, (hv.Element, hv.Layout, hv.Overlay, hv.NdLayout))


def test_taylor_diagram_hvplot():
    df = pd.DataFrame({"obs": [1, 2, 3], "model": [1.1, 1.9, 3.2]})
    plot = TaylorDiagramPlot(df)
    hv_obj = plot.hvplot()
    assert isinstance(hv_obj, (hv.Element, hv.Layout, hv.Overlay, hv.NdLayout))


def test_curtain_hvplot():
    data = np.random.rand(5, 10)
    da = xr.DataArray(
        data,
        dims=["level", "time"],
        coords={"level": [1, 2, 3, 4, 5], "time": np.arange(10)},
    )
    plot = CurtainPlot(da)
    hv_obj = plot.hvplot()
    assert isinstance(hv_obj, (hv.Element, hv.Layout, hv.Overlay, hv.NdLayout))


def test_fingerprint_hvplot():
    df = pd.DataFrame(
        {
            "time": pd.date_range("2023-01-01", periods=100, freq="h"),
            "val": np.random.rand(100),
        }
    )
    plot = FingerprintPlot(df, val_col="val")
    hv_obj = plot.hvplot()
    assert isinstance(hv_obj, (hv.Element, hv.Layout, hv.Overlay, hv.NdLayout))


def test_trajectory_hvplot():
    lon = np.linspace(-120, -70, 10)
    lat = np.linspace(20, 50, 10)
    data = np.random.rand(10)
    time = pd.date_range("2023-01-01", periods=10)
    ts_data = np.random.rand(10)
    plot = TrajectoryPlot(lon, lat, data, time, ts_data)
    hv_obj = plot.hvplot()
    assert isinstance(hv_obj, (hv.Element, hv.Layout, hv.Overlay, hv.NdLayout))


def test_diurnal_error_hvplot():
    df = pd.DataFrame(
        {
            "time": pd.date_range("2023-01-01", periods=100, freq="h"),
            "obs": np.random.rand(100),
            "mod": np.random.rand(100),
        }
    )
    plot = DiurnalErrorPlot(df, obs_col="obs", mod_col="mod")
    hv_obj = plot.hvplot()
    assert isinstance(hv_obj, (hv.Element, hv.Layout, hv.Overlay, hv.NdLayout))


def test_wind_barbs_hvplot():
    ws = np.random.rand(10, 10)
    wdir = np.random.rand(10, 10) * 360

    class Grid:
        def __init__(self):
            self.variables = {
                "LAT": np.zeros((1, 1, 10, 10)),
                "LON": np.zeros((1, 1, 10, 10)),
            }

    plot = WindBarbsPlot(ws, wdir, Grid())
    hv_obj = plot.hvplot()
    assert isinstance(hv_obj, (hv.Element, hv.Layout, hv.Overlay, hv.NdLayout))


def test_wind_quiver_hvplot():
    ws = np.random.rand(10, 10)
    wdir = np.random.rand(10, 10) * 360

    class Grid:
        def __init__(self):
            self.variables = {
                "LAT": np.zeros((1, 1, 10, 10)),
                "LON": np.zeros((1, 1, 10, 10)),
            }

    plot = WindQuiverPlot(ws, wdir, Grid())
    hv_obj = plot.hvplot()
    assert isinstance(hv_obj, (hv.Element, hv.Layout, hv.Overlay, hv.NdLayout))


def test_windrose_hvplot():
    wd = np.random.rand(100) * 360
    ws = np.random.rand(100) * 10
    plot = Windrose(wd=wd, ws=ws)
    hv_obj = plot.hvplot()
    assert isinstance(hv_obj, (hv.Element, hv.Layout, hv.Overlay, hv.NdLayout))


def test_meteogram_hvplot():
    df = pd.DataFrame(
        {"A": [1, 2], "B": [3, 4]}, index=pd.date_range("2023-01-01", periods=2)
    )
    plot = Meteogram(df=df, variables=["A", "B"])
    hv_obj = plot.hvplot()
    assert isinstance(hv_obj, (hv.Element, hv.Layout, hv.Overlay, hv.NdLayout))


def test_upper_air_hvplot():
    lat = np.linspace(20, 50, 5)
    lon = np.linspace(-120, -70, 5)
    hgt = np.random.rand(5, 5)
    u = np.random.rand(5, 5)
    v = np.random.rand(5, 5)
    plot = UpperAir(lat=lat, lon=lon, hgt=hgt, u=u, v=v)
    hv_obj = plot.hvplot()
    assert isinstance(hv_obj, (hv.Element, hv.Layout, hv.Overlay, hv.NdLayout))


def test_profile_hvplot():
    x = np.arange(10)
    y = np.arange(10)
    plot = ProfilePlot(x=x, y=y)
    hv_obj = plot.hvplot()
    assert isinstance(hv_obj, (hv.Element, hv.Layout, hv.Overlay, hv.NdLayout))


def test_facet_grid_hvplot():
    df = pd.DataFrame(
        {"x": [1, 2, 3, 4], "y": [4, 5, 6, 7], "col": ["A", "A", "B", "B"]}
    )
    plot = FacetGridPlot(df, col="col")
    hv_obj = plot.hvplot(x="x", y="y", kind="scatter")
    assert isinstance(hv_obj, (hv.Element, hv.Layout, hv.Overlay, hv.NdLayout))


def test_bivariate_polar_hvplot():
    df = pd.DataFrame({"ws": [1, 2], "wd": [45, 90], "val": [10, 20]})
    plot = BivariatePolarPlot(df, ws_col="ws", wd_col="wd", val_col="val")
    hv_obj = plot.hvplot()
    assert isinstance(hv_obj, (hv.Element, hv.Layout, hv.Overlay, hv.NdLayout))


def test_stick_plot_hvplot():
    u = np.array([1, 2, 3])
    v = np.array([0.5, 1, 1.5])
    y = np.array([10, 20, 30])
    plot = StickPlot(u, v, y)
    hv_obj = plot.hvplot()
    assert isinstance(hv_obj, (hv.Element, hv.Layout, hv.Overlay, hv.NdLayout))


def test_vertical_box_plot_hvplot():
    data = np.random.rand(100)
    y = np.random.rand(100) * 100
    thresholds = [0, 50, 100]
    plot = VerticalBoxPlot(data, y, thresholds)
    hv_obj = plot.hvplot()
    assert isinstance(hv_obj, (hv.Element, hv.Layout, hv.Overlay, hv.NdLayout))
