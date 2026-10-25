"""Compose a Skew-T, spatial field, and pressure profile in one figure."""

import cartopy.crs as ccrs
import numpy as np
import pandas as pd
import xarray as xr

from monet_plots import PlotLayout, SkewTPlot
from monet_plots.plots.profile import VerticalProfilePlot
from monet_plots.plots.spatial_mesh import SpatialMeshPlot

pressure = np.array([1000, 925, 850, 700, 500, 300, 200, 100])
temperature = np.array([24, 18, 13, 2, -18, -42, -54, -68])
dewpoint = temperature - 6
layout = PlotLayout((2, 2), figsize=(13, 9))
layout.fig.set_layout_engine("constrained")

skew = layout.add(
    SkewTPlot,
    cell=(0, 0),
    rowspan=2,
    subplot_kwargs={"projection": "monet_skewt"},
    plot_kwargs={"aspect": "auto"},
)
skew.plot(pressure, temperature, dewpoint, label="Observed", parcel=True)

lat = np.linspace(25, 50, 30)
lon = np.linspace(-125, -65, 45)
lon2d, lat2d = np.meshgrid(lon, lat)
field = xr.DataArray(
    np.sin(np.deg2rad(lat2d * 3)) * np.cos(np.deg2rad(lon2d)),
    dims=("lat", "lon"),
    coords={"lat": lat, "lon": lon},
    name="Temperature anomaly",
)
map_plot = layout.add(
    SpatialMeshPlot,
    field,
    cell=(0, 1),
    subplot_kwargs={"projection": ccrs.PlateCarree()},
    plot_kwargs={"data_crs": ccrs.PlateCarree(), "coastlines": False},
)
map_plot.plot(cmap="coolwarm")

rng = np.random.default_rng(21)
levels = np.repeat([1000, 850, 700, 500, 300, 200], 80)
mean = np.repeat([24, 15, 5, -15, -42, -55], 80)
observed = mean + rng.normal(0, 2, len(levels))
model = observed + 1 + rng.normal(0, 1.5, len(levels))
sounding = pd.DataFrame(
    {"pressure_hpa": levels, "temperature_obs": observed, "temperature_model": model}
)
profile = layout.add(
    VerticalProfilePlot,
    sounding,
    "temperature_obs",
    "temperature_model",
    cell=(1, 1),
    plot_kwargs={
        "alt_col": "pressure_hpa",
        "bins": np.array([100, 200, 300, 500, 700, 850, 1000]),
    },
)
profile.plot()
layout.save("mixed_scientific_layout.png", dpi=140)
layout.close()
