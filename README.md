# MONET Plots

Scientific plotting for geospatial models and observations, built on Matplotlib,
Cartopy, and xarray.

```python
import monet_plots as mp

# field: DataArray with model, season, latitude and longitude dimensions.
result = mp.spatial(field, row="model", col="season", vmin=0, vmax=80)
result.save("comparison.pdf")
```

Spatial facets share a color scale and colorbar by default. Regular, irregular,
and curvilinear grids, projected coordinates, station overlays, and Dask-backed
arrays are supported. See the [spatial plotting guide](docs/spatial-comparisons.md)
and [runnable comparison](examples/plot_spatial_comparison.py).

Install locally with `pip install .`, or use `pip install -e ".[test,docs]"`
for development. Python 3.11+ is required; CI covers 3.11–3.14. Interactive backends use the
`[interactive]` extra.

## Development

### Pre-commit Hooks

This repository uses [pre-commit](https://pre-commit.com/) to ensure code quality and consistency.

To set up pre-commit locally:

1.  Install pre-commit:
    ```bash
    pip install pre-commit
    ```

2.  Install the git hook scripts:
    ```bash
    pre-commit install
    ```

Now, pre-commit will run automatically on `git commit`. You can also run it against all files manually:

```bash
pre-commit run --all-files
```

## Releases

Use Conventional Commit PR titles and squash merge into `main`. After CI passes,
automation maintains a version/changelog release PR; merging it creates a GitHub
release with wheel and source archives. See [release setup and versioning](docs/releases.md)
for the required token and repository settings.

## Model evaluation and benchmarks

Use `mp.compare_spatial(observations, model, row="model")` for paired
observation/model/bias maps with shared concentration scales and a separate
zero-centered bias scale. See the [model evaluation guide](docs/model-evaluation.md).

The [benchmark harness](benchmarks/README.md) measures large NumPy/Dask maps,
facets, and comparisons in isolated processes, including rendering time,
resident memory, and executed Dask tasks.
