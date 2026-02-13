# Facet Grids

Facet grids allow you to visualize data across different subsets of your dataset in a grid of subplots.

## General Facet Grid (Seaborn-based)

The `FacetGridPlot` class uses Seaborn to create grids from pandas DataFrames.

::: monet_plots.plots.facet_grid.FacetGridPlot

## Spatial Facet Grid (Xarray-based)

The `SpatialFacetGridPlot` class uses xarray's FacetGrid to create geospatial grids with cartopy support.

::: monet_plots.plots.facet_grid.SpatialFacetGridPlot

### Example

```python
from monet_plots.plots import SpatialFacetGridPlot

# Create a spatial facet grid
grid = SpatialFacetGridPlot(da, col='time', projection=ccrs.PlateCarree())
grid.plot(plot_func='contourf', coastlines=True)
```
