from importlib.metadata import version as _distribution_version

from .api import compare_spatial, spatial
from .plots.brier_decomposition import BrierScoreDecompositionPlot
from .plots.conditional_bias import ConditionalBiasPlot
from .plots.ensemble import SpreadSkillPlot
from .plots.facet_grid import FacetGridPlot, SpatialFacetGridPlot
from .plots.kde import KDEPlot
from .plots.layout import PlotLayout
from .plots.meteogram import Meteogram
from .plots.performance_diagram import PerformanceDiagramPlot
from .plots.profile import ProfilePlot
from .plots.rank_histogram import RankHistogramPlot
from .plots.reliability_diagram import ReliabilityDiagramPlot
from .plots.rev import RelativeEconomicValuePlot
from .plots.roc_curve import ROCCurvePlot
from .plots.scatter import ScatterPlot
from .plots.scorecard import ScorecardPlot
from .plots.skewt import SkewTPlot
from .plots.spatial import SpatialPlot
from .plots.spatial_bias_scatter import SpatialBiasScatterPlot
from .plots.spatial_comparison import SpatialComparisonPlot
from .plots.spatial_contour import SpatialContourPlot
from .plots.spatial_imshow import SpatialImshowPlot
from .plots.spatial_mesh import SpatialMeshPlot
from .plots.taylor_diagram import TaylorDiagramPlot
from .plots.timeseries import TimeSeriesPlot
from .plots.upper_air import UpperAir
from .plots.wind_barbs import WindBarbsPlot
from .plots.wind_quiver import WindQuiverPlot
from .plots.windrose import Windrose

# src/monet_plots/__init__.py
from .style import get_available_styles, get_style_setting, set_style, style_context

__all__ = [
    "PlotLayout",
    "SkewTPlot",
    "compare_spatial",
    "SpatialComparisonPlot",
    "SpatialMeshPlot",
    "SpatialFacetGridPlot",
    "spatial",
    "style_context",
    "SpatialPlot",
    "TimeSeriesPlot",
    "TaylorDiagramPlot",
    "KDEPlot",
    "ScatterPlot",
    "WindQuiverPlot",
    "WindBarbsPlot",
    "SpatialBiasScatterPlot",
    "SpatialContourPlot",
    "FacetGridPlot",
    "PerformanceDiagramPlot",
    "ROCCurvePlot",
    "ReliabilityDiagramPlot",
    "RankHistogramPlot",
    "BrierScoreDecompositionPlot",
    "ScorecardPlot",
    "RelativeEconomicValuePlot",
    "ConditionalBiasPlot",
    "SpreadSkillPlot",
    "SpatialImshowPlot",
    "Windrose",
    "Meteogram",
    "UpperAir",
    "ProfilePlot",
    "set_style",
    "get_available_styles",
    "get_style_setting",
]

# Read the installed distribution metadata; pyproject.toml owns the version.
__version__ = _distribution_version("monet-plots")
