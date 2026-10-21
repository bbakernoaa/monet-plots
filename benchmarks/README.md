# Spatial rendering benchmarks

Run from the repository checkout after installing the benchmark extra:

```bash
python -m pip install -e '.[benchmark]'
python benchmarks/bench_spatial.py --profile smoke --repeat 1 --output artifacts/smoke.json
python benchmarks/bench_spatial.py --profile standard --repeat 2 --output artifacts/benchmarks.json
python benchmarks/plot_results.py artifacts/benchmarks.json --output artifacts/benchmarks.png
```

`--case NAME` selects a case and can be repeated. The standard profile covers
1024 × 1024 NumPy and Dask fields; four-facet figures; regular rasters versus
meshes; automatic versus explicit bounds; three-column observation/model/bias
figures; four-row comparisons (12 panels); and a 2048 × 2048 mesh. It also compares
row-shared and independent-panel evaluation of comparison data. The smoke profile
uses 32 × 32 grids and exercises the same public APIs without large allocations.

Each repetition runs in a **fresh subprocess**, warms imports/fonts with an 8 × 8
map, and uses Agg, 80 dpi, one numeric thread, and Dask's synchronous scheduler.
Coordinates and fields are deterministic; input fields are float32. Dask cases
generate arrays lazily from 1D coordinate vectors, without constructing the whole
input field in advance. Map features are disabled, so there is no coastline
network access or variable file I/O. This measures rendering costs rather than
NetCDF/Zarr reads, regridding, or application preprocessing.

JSON includes raw repetitions, median stage timings, cells displayed, chunks,
PNG size, package versions, OS/CPU information, and a hash of the plotting sources
and benchmark script. Time stages are input generation, scale selection/artist
construction, canvas draw, and in-memory PNG export (which redraws the figure).
Dask task counts identify which stages execute lazy work. Export measurements are
reported separately from construction plus the first draw.

RSS is sampled with psutil every 10 ms and includes NumPy, Cartopy, Matplotlib,
and other native allocations. Both absolute peak and increase above warmed
imports before input generation are reported. Sampling may miss short-lived
peaks, and allocator behavior affects measurements. These are measured resident
peaks, not Python-only tracemalloc results or formal memory upper bounds.

Timings and RSS are machine-specific. Two repetitions provide a small sample,
not a statistical confidence interval. Compare like-for-like runs on the same
hardware and software. CI records weekly/manual measurements as artifacts; it
uses no universal pass/fail timing threshold on shared runners. A failed case
leaves partial JSON/Markdown with the already completed cases.

Matplotlib retains the displayed artists. Four 1024² panels mean 4.2 million
cells; a four-row three-column comparison displays 12.6 million cells, even when
the source array is lazy. This is not an out-of-core figure. Subset, aggregate,
or coarsen deliberately before plotting when the display has far fewer pixels
than source cells. Supply meaningful explicit limits to avoid global reductions;
use imshow for regular grids and pcolormesh for irregular/curvilinear grids.

The checked-in reference report under `results/` records one development-machine
run; rerun the commands to assess your environment.
