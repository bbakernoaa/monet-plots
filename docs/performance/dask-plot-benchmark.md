# Dask plot materialization benchmark

The static time-series plot computes the reduced mean and standard deviation together before passing them to Matplotlib. The spatial bias plot computes bias and the native coordinate arrays together, then broadcasts coordinates as NumPy views just before selecting markers. The map still draws every point and uses the same bias limits and missing-data markers.

## Method

The benchmark builds deterministic synthetic Dask arrays, then measures from the call to `plot()` through the first Agg canvas draw. It excludes imports and figure construction. Dask uses its normal graph optimization and a two-thread local scheduler. A callback counts scheduler runs and executed tasks. Linux peak RSS covers the process, including imports and rendering.

The spatial cases use a 512 × 768 bias grid with 128 × 128 chunks. One case uses NumPy coordinate arrays and the other uses chunked coordinates. Both render without downloading map boundaries. The time series reduces 168 × 256 × 256 values in chunks of 24 × 64 × 64 along the two spatial dimensions.

Run each case in a fresh process, with MONET Plots installed in the environment.

```sh
for attempt in 1 2 3; do
  for plot in spatial spatial-chunked-coordinates timeseries; do
    OPENBLAS_NUM_THREADS=1 python benchmarks/benchmark_dask_plots.py "$plot"
  done
done
```

To compare with the pre-change code, export commit `9857444` to a temporary directory with `git archive`. Set `PYTHONPATH` to that directory's `src` and run the same script from the current checkout.

## Results

Medians of three fresh Linux processes per case, using Python 3.12, Dask 2026.8.0, xarray 2026.9.0, and Matplotlib 3.11.2. Times include rendering.

| Workload | Scheduler runs | Executed tasks | First render (s) | Peak RSS (MiB) |
| --- | ---: | ---: | ---: | ---: |
| Spatial bias, eager coordinates | 1 → 1 | 73 → 73 | 3.256 → 2.934 | 276.2 → 278.3 |
| Spatial bias, Dask coordinates | 1 → 1 | 91 → 75 | 2.917 → 2.990 | 276.6 → 278.4 |
| Time series | 4 → 1 | 1362 → 415 | 1.095 → 0.432 | 197.0 → 196.7 |

The largest improvement was in the time series, where the scheduler ran once and could reuse shared work. Native coordinate computation reduced scheduler tasks for maps with Dask coordinates. These runs showed little change in process peak RSS. Map drawing still allocates the scatter markers, and the timings cover synthetic in-memory data on a local scheduler.
