"""
Observation, model and bias maps
===============================

Two models evaluated against one gridded observational reference. Concentration
maps share a common scale; model-minus-observation bias has a separate symmetric
scale centered on zero. The missing reference region remains masked in the bias.
Fields must already share units, grid coordinates and sampling times.
"""

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import monet_plots as mp
import numpy as np
import xarray as xr

lon = np.linspace(-125, -65, 101)
lat = np.linspace(25, 50, 61)
xx, yy = np.meshgrid(lon, lat)
reference = 35 + 12 * np.cos((xx + 95) / 18) * np.sin((yy - 20) / 15)
observations = xr.DataArray(
    reference,
    dims=("lat", "lon"),
    coords={"lat": lat, "lon": lon},
    name="ozone",
    attrs={"units": "ppb", "long_name": "Surface ozone"},
)
models = xr.concat(
    [
        observations + 6 * np.sin((xx + 100) / 15),
        observations - 4 * np.cos((yy - 35) / 10),
    ],
    dim=xr.IndexVariable("model", ["Model A", "Model B"]),
)
observations = observations.where(
    ~((observations.lon < -115) & (observations.lat > 40))
)
result = mp.compare_spatial(
    observations.chunk({"lat": 20}),
    models.chunk({"model": 1, "lat": 20}),
    row="model",
    projection=ccrs.LambertConformal(central_longitude=-95),
    extent=(-125, -65, 24, 51),
    vmin=0,
    vmax=60,
    bias_limit=8,
    coastlines=True,
    resolution="50m",
    rasterized=True,
    gridlines={
        "draw_labels": True,
        "linewidth": 0.3,
        "alpha": 0.4,
        "xlabel_style": {"size": 8},
        "ylabel_style": {"size": 8},
    },
    size=2.8,
    aspect=1.3,
    style="paper",
)
result.fig.suptitle("Surface ozone: observations, model predictions and bias")
plt.show()
# result.bias and result.paired_counts remain xarray objects for inspection.
# result.save('observation-model-bias.pdf')
