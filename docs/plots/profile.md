# Profile Plots

## ProfilePlot

::: monet_plots.plots.profile.ProfilePlot

## VerticalProfilePlot

::: monet_plots.plots.profile.VerticalProfilePlot

## VerticalBoxPlot

::: monet_plots.plots.profile.VerticalBoxPlot

## VerticalSlice

::: monet_plots.plots.profile.VerticalSlice

### Example

```python
import numpy as np
from monet_plots.plots import VerticalSlice

# Create sample data
x = np.linspace(0, 10, 100)
y = np.linspace(0, 10, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(X) * np.cos(Y)

# Create the plot
plot = VerticalSlice(x=X, y=Y, z=Z)
plot.plot()
```

## StickPlot

::: monet_plots.plots.profile.StickPlot

### Example

```python
import numpy as np
from monet_plots.plots import StickPlot

# Create sample data
u = np.random.rand(10)
v = np.random.rand(10)
y = np.arange(10)

# Create the plot
plot = StickPlot(u, v, y)
plot.plot()
```

## VerticalProfilePlot with pressure coordinates

Use `alt_col` to select pressure instead of height. Pressure is detected from the
column name (for example, `pressure` or `pressure_hpa`) or Xarray units metadata,
and the vertical axis is inverted so larger pressure values appear lower.

```python
plot = VerticalProfilePlot(
    sounding_dataframe,
    obs_col="temperature_obs",
    mod_cols="temperature_model",
    alt_col="pressure_hpa",
    bins=[100, 200, 300, 500, 700, 850, 1000],
)
plot.plot()
```

For coordinates whose names or units do not identify pressure, pass
`vertical_axis="pressure"`. Set `vertical_axis="height"` to force height orientation.
