# Large-dataset measurements

The repository benchmark harness measures complete spatial rendering, including
actual canvas draws, rather than timing only construction of a lazy array.
It covers regular maps, facets, observation/model/bias comparisons, and NumPy and
Dask inputs. Each repetition runs in a fresh process with one numeric thread.

From a repository checkout:

```bash
python -m pip install -e '.[benchmark]'
python benchmarks/bench_spatial.py --profile standard --repeat 2 --output artifacts/benchmarks.json
python benchmarks/plot_results.py artifacts/benchmarks.json --output artifacts/benchmarks.png
```

For a quick API check, use `--profile smoke --repeat 1`. Select individual cases
with repeated `--case NAME` arguments. Read `benchmarks/README.md` for the complete
methodology; raw reference measurements are in `benchmarks/results/`.

## What is measured

- Input construction, scale selection and artist construction, canvas drawing,
  and in-memory PNG export are timed separately. PNG export performs another draw.
- Process resident memory is sampled every 10 ms, including native allocations
  from NumPy, Cartopy and Matplotlib. Absolute peak and increase above warmed
  imports before input generation are reported.
- Dask task completions are counted during construction, drawing and export.
- Reports record package versions, system details, array/chunk sizes, a source
  hash, every repetition, and medians. The weekly/manual performance workflow
  retains JSON, Markdown and chart artifacts for 90 days.

The fields are synthetic float32 data. Dask fields are generated lazily from 1D
coordinates. Features and file/network I/O are disabled. These results do not
measure NetCDF/Zarr read performance, regridding, remote storage, or expensive
application preprocessing. A sampled RSS maximum is not a formal memory upper
bound. Shared runners and short repeat counts do not justify universal timing
thresholds or statistical confidence claims.

## Choosing a workflow

Use `kind="imshow"` for regularly spaced 1D grids when appropriate; meshes are
more general but require cell geometry. Use `pcolormesh` for irregular and
curvilinear coordinates. The library preserves pixel placement and rejects grids
that cannot be represented by a regular image.

Explicit, scientifically meaningful limits avoid global min/max reductions.
That reduces Dask work but does not necessarily dominate total rendering time.
The mesh renderer materializes each displayed panel once before calling
Cartopy/Matplotlib, preventing repeated coercions from rerunning the same graph.
Repeated canvas draws and exports reuse the concrete artists.

For observation/model/bias figures, `materialize="row"` shares source evaluation
across the three panels. `materialize="panel"` evaluates them independently. The
number of optimized Dask tasks is **not** a count of source reads: sharing graphs
can reduce expensive source evaluation while increasing task counts because
fusion changes. Row sharing may not improve time or memory for cheap synthetic
arrays; benchmark your actual storage/computation workload.

Dask defers the input computation, but the resulting Matplotlib figure holds its
displayed cells in memory. Four 1024² maps display 4.2 million cells; a four-row,
three-column comparison displays 12.6 million cells. Subset and aggregate
explicitly when a figure has far fewer display pixels than source cells. Choose
area/time/sampling weights appropriate to the science. Rasterizing PDF/SVG layers
helps output size; it does not eliminate in-memory geometry.

Close figures after export and release references to plot objects when finished.
Explicitly persisting entire Dask cubes can help repeated analyses but increases
memory use; this library does not persist an entire dataset implicitly.

## Reference results

The checked-in reference report records a development-machine run, not a speed
or memory guarantee. Reproduce it with the commands above and compare on the
same hardware and dependency versions.

Measurements on this Linux/Python 3.12.14 development environment (two isolated
repetitions; medians, float32 fields):

| Case | Displayed cells | Render time | Absolute peak RSS |
| --- | ---: | ---: | ---: |
| Single 1024² Dask mesh | 1,048,576 | 0.45 s | 278 MiB |
| Four 1024² meshes, explicit limits | 4,194,304 | 1.76 s | 427 MiB |
| Four 1024² images, explicit limits | 4,194,304 | 0.37 s | 268 MiB |
| Four-row observation/model/bias comparison | 12,582,912 | 5.12 s | 824 MiB |
| Single 2048² Dask mesh | 4,194,304 | 1.68 s | 568 MiB |

Render time excludes the additional PNG export; see the raw JSON/Markdown for
all stages, repeats and environment details. On this synthetic regular grid,
four images rendered about 4.8 times faster than four meshes and used about
159 MiB less peak RSS. This is a renderer choice, not evidence that images can
represent an irregular grid correctly.

Explicit limits reduced four-facet mesh construction tasks from 632 to 260,
without a measured total rendering-time improvement in this small sample.
Independent-panel comparison tasks were 99 with explicit limits, versus 131
with row sharing, while a delayed-source regression test verifies one source
evaluation per shared row instead of three. Cheap synthetic data did not show
a timing benefit from row sharing; test real storage/computation workloads before
choosing a strategy for performance reasons.

The benchmark investigation also fixed two issues: regular float32 raster
coordinates were rejected by an overly strict spacing tolerance, and meshes
coerced lazy panel values twice. The mesh boundary now computes a panel once;
its draws and exports execute no further Dask tasks in these measurements.
