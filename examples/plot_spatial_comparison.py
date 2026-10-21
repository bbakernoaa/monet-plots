"""
Model and season spatial comparison
==================================

Two models across four seasons, with one physical color scale and station
observations selected by season. The deterministic synthetic data makes this
example runnable without downloading a scientific dataset. Coastlines use
Cartopy's Natural Earth cache (downloaded on first rendering).
"""

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import monet_plots as mp
import numpy as np
import xarray as xr

lon = np.linspace(-125, -65, 61)
lat = np.linspace(25, 50, 31)
xx, yy = np.meshgrid(lon, lat)
base = 30 + 15 * np.cos((xx + 95) / 18) * np.sin((yy - 20) / 15)
values = np.stack(
    [
        np.stack([base + seasonal + model * 5 for seasonal in [-10, 0, 15, -2]])
        for model in range(2)
    ]
)
ozone = xr.DataArray(
    values,
    dims=("model", "season", "lat", "lon"),
    coords={
        "model": ["Model A", "Model B"],
        "season": ["DJF", "MAM", "JJA", "SON"],
        "lat": lat,
        "lon": lon,
    },
    name="ozone",
    attrs={"long_name": "Surface ozone", "units": "ppb"},
).chunk({"model": 1, "season": 1})
observations = xr.DataArray(
    [[20, 35, 30, 25], [30, 40, 35, 30], [45, 55, 50, 40], [30, 35, 32, 28]],
    dims=("season", "site"),
    coords={
        "season": ozone.season,
        "site": range(4),
        "lon": ("site", [-115, -100, -85, -75]),
        "lat": ("site", [35, 40, 33, 42]),
    },
    name="ozone",
    attrs={"units": "ppb"},
)
result = mp.spatial(
    ozone,
    row="model",
    col="season",
    kind="pcolormesh",
    projection=ccrs.LambertConformal(central_longitude=-95),
    data_crs=ccrs.PlateCarree(),
    extent=(-125, -65, 24, 51),
    vmin=0,
    vmax=80,
    cmap="viridis",
    observations=observations,
    panel_labels=True,
    coastlines=True,
    resolution="50m",
    gridlines={
        "draw_labels": True,
        "linewidth": 0.3,
        "alpha": 0.4,
        "xlabel_style": {"size": 8},
        "ylabel_style": {"size": 8},
    },
    size=2.8,
    aspect=1.2,
    rasterized=True,
    style="paper",
    cbar_kwargs={
        "orientation": "horizontal",
        "pad": 0.08,
        "fraction": 0.04,
        "label": "Surface ozone (ppb)",
    },
)
for ax, (_, _, selection) in zip(result.axes.flat, result._panels()):
    ax.set_title(f"{selection['model']} · {selection['season']}", fontsize=10)
result.fig.suptitle("Surface ozone: model and seasonal comparison")
plt.show()
# Export with result.save("ozone-comparison.pdf") or result.save("ozone-comparison.png").
