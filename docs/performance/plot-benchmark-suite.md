# Plot benchmark coverage and optimizations

The static suite inventories 43 concrete plot types, including the low-level Taylor diagram and categorical function. It also covers six public Cartopy helpers. It uses shipped examples where available and supplies deterministic fixtures for the other cases. All 49 cases rendered. BasePlot is infrastructure, SpScatterBiasPlot is an unresolved export, and categorical_timeseries is an unimplemented placeholder.

The Dask suite exercises separately the workloads that can use lazy data. It records scheduler runs, task executions, wall time, and process peak RSS for construction, plot preparation, and canvas drawing. It covers automatic and fixed scales or bins, chunked labels, multiple models, contour and mesh modes, and faceting. The separate Dask benchmark covers raw time series, spatial bias, and wind grids.

## Reproduce

Use an installed MONET Plots test environment on Linux. The static map examples use Cartopy's Natural Earth coastline and state boundary files. Pass the path to an existing Cartopy cache for offline runs.

```sh
python benchmarks/benchmark_plot_types.py --cartopy-data /path/to/cartopy --output static-plot-types.csv

export PYTHONPATH="$PWD/src"
export OPENBLAS_NUM_THREADS=1
for case in $(python benchmarks/benchmark_lazy_plot_types.py --list); do
  python benchmarks/benchmark_lazy_plot_types.py "$case"
done > lazy-plot-workloads.jsonl

for case in wind-barbs wind-quiver; do
  python benchmarks/benchmark_dask_plots.py "$case"
done
```

The static benchmark defaults to three fresh subprocesses per case. The 49-case coverage sweep used `--preload --repeats 1` to fork one isolated Linux process per case after loading imports. It excludes output encoding and disk writes. Use the same case and mode when comparing revisions. The estimate of peak RSS includes the Python process, the data, and renderer allocations.

For a pinned baseline, extract commit `814ca5b` with `git archive`. Use its `src` as `--source-root` for the static benchmark or `PYTHONPATH` for the Dask benchmarks. Keep the benchmark script and inputs the same.

## What changed

- Wind maps thin the grid before materialization and share one compute call for selected speeds, directions, and coordinates.
- Time-series bias, MAE, and RMSE use grouped means of pointwise errors when the xarray dimensions match. Each valid observation/model pair contributes equally within its bin. Error-bar series compute means and spreads together.
- Spatial track and contour plots reuse one snapshot of render data for extents or scale and for drawing. Spatial facets reuse Xarray's full-panel snapshot for the shared color scale and rendering, then restore the lazy source.
- Radar metrics reuse the shared observation array across sequential model comparisons. Diurnal and conditional bias resolve chunked group labels without converting the measurements to pandas.
- Ridgeline gradients draw each ridge's colored segments as one collection. Automatic range uses a joint minimum/maximum reduction. Exact KDE still reads one ridge at a time.

## Measured changes

| Dask workload | Scheduler runs | Executed tasks | Plot stage (s) |
| --- | ---: | ---: | ---: |
| Three-model daily RMSE | 3 → 1 | 13,587 → 262 | 44.742 → 1.134 |
| Three error-bar series | 6 → 1 | 1,049 → 713 | 0.718 → 0.297 |
| Spatial track | 9 → 1 | 129 → 25 | 0.164 → 0.065 |
| Spatial contour with automatic levels | 2 → 1 | 173 → 65 | 0.195 → 0.119 |
| Four spatial raster panels | 5 → 1 | 485 → 161 | 0.294 → 0.143 |
| Four spatial contour panels | 5 → 1 | 485 → 161 | 0.539 → 0.317 |
| Ridgeline, automatic range, solid fill | 10 → 9 | 224 → 192 | 0.253 → 0.289 |

Each row above is a paired workload comparison from fresh processes. These single-run wall times are indicative. Task counts measure scheduler work independently of wall-time variation. The daily RMSE run used about 6% more peak process memory after the change, so large-model memory still needs monitoring.

| Render benchmark | Median first render (s) | Median process peak RSS (MiB) |
| --- | ---: | ---: |
| Dask wind barbs | 1.640 → 0.460 | 284.8 → 207.5 |
| Dask wind arrows | 1.616 → 0.489 | 284.5 → 207.8 |
| Static ridgeline gradients | 2.055 → 0.216 | 235.8 → 222.1 |

The render comparison uses three fresh processes per side. Both wind cases went from two scheduler runs and 3,074 tasks to one run and 1,026 tasks. The wind clock includes construction because the baseline computed both input fields there.

## Remaining workload limits

The suite exercises representative static and local Dask workloads. Large remote or distributed datasets, alternate chunk layouts, and interactive backends have their own costs. Table-oriented plots accept prepared tables, so aggregate gridded inputs before invoking them. Adaptive bins in conditional bias and wind rose require discovering boundaries before binning. Explicit edges skip that pass when they match the scientific definition. Dense scatter and trajectory plots can be dominated by drawing markers. Spatial facets still need the panel array in memory for Xarray's shared scale.

## Chunked input compatibility

Run the offline eager-versus-chunked check with the test extras installed.

```bash
python benchmarks/check_chunked_plot_parity.py --output chunked-plot-parity.jsonl
```

The check passes Dask-backed xarray objects, Dask DataFrames, or Dask arrays
through the same public entry points as the eager examples. It compares the
first rendered Agg figures and reports Dask scheduler runs. The standalone
`SpatialPlot` map builder and low-level `TaylorDiagram` accept map settings or
scalar statistics. Their data-aware variants are covered by the same suite.
Static map features are bypassed so the check can run offline.

Some table-based plots materialize Dask inputs in pandas for grouping or reshaping. Dask DataFrame input
requires the `dask[dataframe]` extra. Use xarray for labeled chunked fields and
the native xarray reduction paths. Raw reliability diagrams reduce forecasts to
per-group bin counts and sums before computing the combined summary.

The `reliability-xarray` and `reliability-dataframe` workloads benchmark raw forecasts.
In a local raw reliability run with 1,048,576 samples, four groups, and ten
bins, the chunked xarray path went from three scheduler runs to one. Peak
process RSS went from about 310 MiB to 247 MiB, and plot plus first draw took
0.86 s and 0.76 s respectively. These are single-run measurements.
