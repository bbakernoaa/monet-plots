# Heterogeneous Plot Layouts

`PlotLayout` arranges different MONET-PLots classes in one Matplotlib figure. Use
it for summaries that combine unlike panels, such as a sounding, a map, and a
vertical profile. Use `FacetGridPlot` when repeating one kind of plot across
data subsets.

## Example: a Skew-T with two companion panels

```python
import cartopy.crs as ccrs
from monet_plots import PlotLayout, SkewTPlot
from monet_plots.plots.profile import VerticalProfilePlot
from monet_plots.plots.spatial_mesh import SpatialMeshPlot

layout = PlotLayout((2, 2), figsize=(13, 9))
layout.fig.set_layout_engine("constrained")

sounding = layout.add(
    SkewTPlot,
    cell=(0, 0),
    rowspan=2,
    subplot_kwargs={"projection": "monet_skewt"},
    plot_kwargs={"aspect": "auto"},
)
sounding.plot(pressure, temperature, dewpoint, parcel=True)

map_plot = layout.add(
    SpatialMeshPlot,
    temperature_map,
    cell=(0, 1),
    subplot_kwargs={"projection": ccrs.PlateCarree()},
    plot_kwargs={"data_crs": ccrs.PlateCarree()},
)
map_plot.plot(cmap="coolwarm")

profile = layout.add(
    VerticalProfilePlot,
    sounding_dataframe,
    "temperature_obs",
    "temperature_model",
    cell=(1, 1),
    plot_kwargs={"alt_col": "pressure_hpa", "bins": 10},
)
profile.plot()
layout.save("sounding_summary.png", dpi=180)
layout.close()
```

`cell` is a zero-based `(row, column)` pair. `rowspan` and `colspan` let a panel
span grid cells. `subplot_kwargs` are sent to Matplotlib when creating the axes,
so use them for Cartopy or Skew-T projections. `plot_kwargs` go to the plot
constructor; additional positional arguments go to that constructor as well.
Call the returned object's `plot()` method to draw data. Set `aspect="auto"` for
a Skew-T that should fill a tall or wide layout cell; the default `80.5` keeps
the standard standalone diagram proportions.

::: monet_plots.plots.layout.PlotLayout
