# Observation, model and bias maps

`compare_spatial` creates three columns: observations, model values, and **model
minus observations**. Observations and model values share one scale; bias uses a
separate diverging scale centered on zero. Add `row="model"` or `row="season"`
for repeated comparisons.

```python
import cartopy.crs as ccrs
import monet_plots as mp

# DataArrays already collocated on the same grid, in the same physical units.
# observations: lat × lon; predictions: model × lat × lon.
result = mp.compare_spatial(
    observations, predictions,
    row="model",
    projection=ccrs.LambertConformal(),
    extent=(-125, -65, 24, 51),
    vmin=0, vmax=60, bias_limit=8,
    cmap="viridis", bias_cmap="RdBu_r",
    rasterized=True,
)
result.save("model-evaluation.pdf")
```

The [runnable example](gallery/plot_observation_model_bias.md) uses synthetic
surface ozone and a missing observation region. It needs no scientific data
download; Cartopy caches the coastline on first use.

## Pairing and interpretation

Inputs are xarray DataArrays. Select Dataset variables explicitly. Field dimension
names, sizes, index labels, and spatial coordinates must match exactly. For
curvilinear grids, the 2D longitude and latitude coordinates must also match.
Where dimensions lack explicit index labels, they pair by position; the spatial
coordinates must still match. Supply explicit period/model labels to make those
comparisons unambiguous. Only a model-only `row` dimension broadcasts static
observations intentionally.
Use `kind="scatter"` for collocated station pairs on a shared sample dimension.

Model output and station observations must be collocated **before** calling this
function. Regrid/interpolate explicitly, choose a scientifically justified method,
and align averaging windows and vertical levels. This function does not infer
representativeness, collocation, unit conversion, or statistical significance.

Both inputs must have the same `units` attribute, or both must omit it. Supplying
units on only one input raises an error. Unit strings are compared literally:
convert equivalent units and normalize their labels beforehand. `data_crs` applies
to both inputs; projected coordinates require an explicit CRS.

Non-finite values remain masked. A bias is shown only where both inputs are finite.
The observation/model panels retain their individual valid values, even where a
pair is missing. The returned `bias`, `paired`, and `paired_counts` expose those
choices as xarray objects; counts are per row and have dimensionless units.
Differences are calculated as **model − observations**, never the reverse.

## Scales and performance

`vmin` and `vmax` set the common observation/model bounds. `bias_limit=8` sets
bias bounds to −8 and +8. Without limits, the common bounds cover finite values
from both inputs and bias limits use the largest absolute finite paired difference
across all rows. All-zero bias uses ±1 to retain a valid zero-centered scale.

A custom concentration `norm` is supported. For bias, a custom `bias_norm` must
be a `TwoSlopeNorm` centered on zero; do not combine it with `bias_limit`.
Use `cbar_kwargs` for shared colorbar options and `bias_cbar_kwargs` to override
bias-specific options. Set `colorbar=False` to omit both bars.

Automatic bounds are reduced in one shared Dask compute graph. Explicit bounds
skip those reductions. By default, `materialize="row"` evaluates the three fields
in one shared graph per row, avoiding repeated source work without persisting an
entire cube. `materialize="panel"` evaluates panels independently for comparison
or different memory tradeoffs. Temporary arrays and Matplotlib artists still
consume memory; reducing the number of tasks does not guarantee lower peak RSS.

`SpatialComparisonPlot(...).plot()` offers the same workflow with delayed
rendering. Construction preserves lazy field values and checks coordinates;
validation may evaluate lazy coordinate arrays. The object can be rendered once;
create a new object for another configuration.

```python
print(result.paired_counts.compute())
# Inspect bias directly with xarray:
print(result.bias.mean().compute())
```

Do not use unweighted spatial means for quantitative conclusions when grid-cell
areas or sampling weights differ. Aggregation and weighting remain explicit
analysis steps.

::: monet_plots.plots.spatial_comparison.SpatialComparisonPlot
