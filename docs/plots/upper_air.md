# Upper Air

The `UpperAir` class provides specialized spatial visualizations for upper-air meteorological data, such as pressure level charts.

## Overview

Upper-air plots often include geopotential height contours, wind barbs, and temperature shading for specific pressure levels (e.g., 500 hPa).

## Classes

::: monet_plots.plots.upper_air.UpperAir

## Skew-T soundings

`SkewTPlot` draws atmospheric soundings on a native Matplotlib skewed-temperature,
log-pressure diagram. It uses the project's existing NumPy, SciPy, and Matplotlib
dependencies.

```python
from monet_plots import SkewTPlot

plot = SkewTPlot()
plot.plot(
    pressure_hpa,
    temperature_c,
    dewpoint_c,
    u=u_wind,
    v=v_wind,
    label="Observed",
    parcel=True,
)
plot.save("sounding.png")
```

Unlabeled arrays use hPa, Celsius, and m/s. DataArray units are read from their
`units` attributes; explicit `pressure_units`, `temperature_units`, and
`wind_units` override them. Parcel ascent marks the LCL and uses a liquid-water
pseudoadiabatic approximation above it; CAPE and CIN are not calculated.
The diagram uses a standard Skew-T aspect ratio of 80.5; pass `aspect=` to
`SkewTPlot` to adjust it for your layout. In a grid cell spanning multiple rows,
use `aspect="auto"` so the Skew-T fills the allocated height. Multiple calls to
`plot` overlay labeled
soundings. For subplot layouts, create
axes with `subplot_kw={"projection": "monet_skewt"}` and pass an axis to
`SkewTPlot`.
