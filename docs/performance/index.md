# Performance

For measured rendering time, native memory use, and large NumPy/Dask datasets,
see [Large-dataset measurements](large-datasets.md). The repository includes an
isolated-process benchmark harness and a weekly/manual performance workflow.

## Practical starting points

- Select the required times, levels and region before plotting.
- Use meaningful explicit color limits to avoid global scale reductions.
- Use images for regular grids and meshes for irregular or curvilinear grids.
- Aggregate explicitly when the source has many more cells than the display has pixels.
- Close figures and release plot objects after exporting.

```python
import monet_plots as mp

# field is an xarray DataArray on lat/lon dimensions.
# Select aggregation and weighting appropriate to your scientific question.
coarse = field.coarsen(lat=4, lon=4, boundary="trim").mean()
result = mp.spatial(coarse, vmin=0, vmax=80)
result.save("overview.png")
result.close()
```

For common plotting options, see [Spatial comparisons](../spatial-comparisons.md)
and [Observation/model/bias maps](../model-evaluation.md).
