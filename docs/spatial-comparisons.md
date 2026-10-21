# Spatial maps and scientific comparisons

Use `monet_plots.spatial` for single maps and facets. The existing plot classes
remain available. The default renderer is `pcolormesh`, supporting independent
1D coordinates and 2D curvilinear grids.

```python
import cartopy.crs as ccrs
import monet_plots as mp

# ozone is an xarray.DataArray with dimensions model, season, lat, lon.
result = mp.spatial(
    ozone,
    row="model", col="season",
    projection=ccrs.LambertConformal(),
    data_crs=ccrs.PlateCarree(),
    extent=(-125, -65, 24, 51),
    cmap="viridis", vmin=0, vmax=80,
    colorbar="shared", panel_labels=True,
    rasterized=True,
)
result.save("ozone-comparison.pdf")
result.save("ozone-comparison.png", dpi=300)
```

See the [runnable model × season example](gallery/plot_spatial_comparison.md).
It includes seasonal station observations and a shared, horizontal colorbar.

## Coordinates and projections

`projection` is the display projection. `data_crs` describes the plotted
coordinates; the default is geographic PlateCarree. Projected coordinates in
metres require the corresponding Cartopy CRS. CF `standard_name`, degree units,
and axis metadata are recognized before exact `lon`/`lat`, `longitude`/`latitude`,
or `x`/`y` names. Ambiguous coordinates require explicit `x=` and `y=` overrides.
Coordinate names alone never establish a projected CRS.

`extent` sets the viewport in `extent_crs`, which defaults to longitude/latitude,
even when the plotted data is projected. `imshow` uses separate `image_extent`
bounds in data coordinates. Otherwise regular coordinate centers determine its
pixel edges. Irregular or curvilinear grids should use `pcolormesh` or `contourf`.

Descending independent grid axes are reordered together with their data. Station
pairs and curvilinear coordinates retain their original pairing. Wrapped 1D
longitudes are unwrapped without modifying the input. For a region crossing the
dateline, use `projection=ccrs.PlateCarree(central_longitude=180)` and an extent
in an appropriate `extent_crs`; a projection seam can split a map at its boundary.
Both 0–360 and −180–180 longitude conventions can be plotted.

## Color scales and missing values

Facets use one normalization and one actual mappable colorbar by default. Use
`colorbar=False` to retain comparable scales without a colorbar, or `colorbar="each"`
to use independent panel scales. The legacy `add_colorbar` boolean remains supported
by `map_monet()` and spatial plotters.

Explicit `vmin`/`vmax`, `norm` (including `LogNorm` and `TwoSlopeNorm`), colormap/norm
tuples, and contour level sequences are supported. Supply either a norm or bounds.
Contour colorbars reflect the actual contour levels, including unequal intervals.
Missing values remain masked. Entirely missing datasets require explicit scale
limits; individual empty panels work with the shared scale from other panels.

For a Dataset, select `var_name="ozone"` or facet with `col="variable"`.
Variables with different units require `colorbar="each"` or prior unit conversion.
Matching unit strings do not establish scientific comparability: choose sensible
variables and explicit scales for your scientific question. No automatic unit
conversion, statistical aggregation, regridding, or significance inference occurs.

## Observations and customization

Pass `observations=...` and, for Dataset/DataFrame inputs, `obs_var="ozone"`.
Observations use each panel's normalization. Matching facet dimensions/columns are
selected automatically; ungrouped points repeat across panels. `obs_crs` defaults
to geographic coordinates independently of model coordinates. Known incompatible
unit strings raise an error. Convert units explicitly before plotting.

The returned plot exposes `fig`, `axes`, `artists`, `colorbars`, `save()`, and
`close()`. Spatial facet grids also expose `plotters`, `mappables`, and `overlays`.
Customize them with ordinary Matplotlib. An existing Cartopy `ax=` can be supplied
for a single map; it cannot be combined with facets.

```python
from monet_plots import SpatialContourPlot

# Existing class constructors now remember their renderer during faceting.
comparison = SpatialContourPlot(ozone, row="model", col="season")
comparison.plot(cmap="viridis", levels=[0, 10, 20, 40, 80])
```

## Styles, performance and installation

Plot construction, rendering and saving apply their named style temporarily.
Use `mp.set_style("paper")` for an intentional session-wide style, or
`with mp.style_context("paper"):` for a temporary scope. Existing class return
values remain unchanged; the convenience API returns the plot object.

Plot-owned xarray objects copy metadata without copying or computing their lazy
arrays. Rendering computes only selected panels and, when necessary, global min/max
reductions for a shared scale. Explicit bounds avoid those global reductions.
Rasterize dense meshes in PDF/SVG output to keep file sizes manageable. Exact
quantiles and unit conversion remain explicit preprocessing steps.

Python 3.11+ is required; CI exercises Python 3.11–3.14 on Linux, macOS and Windows.
Static plotting needs the core dependencies; interactive plotting is available via
`pip install 'monet-plots[interactive]'`. Development uses `[test,docs]`.
The `monet-stats` Git dependency is pinned to an immutable commit for reproducible
installs; replacing it with a versioned package release remains a packaging follow-up.
Cartopy downloads Natural Earth features on first draw. Pre-cache these on offline
systems or use `coastlines=False` and omit other downloaded map features.
