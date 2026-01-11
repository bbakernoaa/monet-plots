"""
Generates a synthetic xarray.Dataset for use in the example gallery.
"""
import numpy as np
import pandas as pd
import xarray as xr


def create_dataset() -> xr.Dataset:
    """
    Creates a synthetic xarray.Dataset for demonstrating plotting functions.

    The dataset is designed to be a minimal but realistic representation of
    atmospheric and air composition data, including temperature, wind, and
    ozone concentration on a latitude-longitude grid with multiple pressure
    levels and time steps.

    Returns
    -------
    xarray.Dataset
        A dataset containing synthetic atmospheric data.
    """
    # Define Coordinates
    lats = np.arange(20, 51, 2.5)
    lons = np.arange(-120, -69, 5)
    levels = np.array([1000, 850, 500, 250])
    times = pd.date_range("2023-01-01T00:00", periods=4, freq="6H")

    # Create coordinate arrays
    n_lats, n_lons, n_levels, n_times = (
        len(lats),
        len(lons),
        len(levels),
        len(times),
    )

    # Generate synthetic data variables
    # Use meshgrid to create arrays that vary realistically across dimensions
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Temperature (K): Decreases with height (pressure level) and varies spatially
    temp_base = 288 - (levels[:, None, None] - 1000) * 0.05
    temp_spatial_var = 5 * (
        np.sin(np.deg2rad(lat_grid)) * np.cos(np.deg2rad(lon_grid))
    )
    temp = temp_base[:, :, :] + temp_spatial_var[None, :, :]

    # Add a time variation
    time_var = 2 * np.sin(np.linspace(0, 2 * np.pi, n_times))
    temp_all = temp[None, :, :, :] + time_var[:, None, None, None]

    # Wind components (m/s)
    u_wind = 10 * np.sin(np.deg2rad(lat_grid))
    v_wind = 10 * np.cos(np.deg2rad(lon_grid))
    u_wind_all = np.tile(u_wind, (n_times, n_levels, 1, 1))
    v_wind_all = np.tile(v_wind, (n_times, n_levels, 1, 1))

    # Ozone (ppbv): Higher concentration in the "stratosphere" (upper levels)
    ozone_base = 20 + (1000 - levels[:, None, None]) * 0.1
    ozone_spatial_var = 10 * (
        np.sin(np.deg2rad(lat_grid * 2)) * np.cos(np.deg2rad(lon_grid * 2))
    )
    ozone = ozone_base[:, :, :] + ozone_spatial_var[None, :, :]
    ozone_all = np.tile(ozone, (n_times, 1, 1, 1))

    # Create the Dataset
    ds = xr.Dataset(
        data_vars={
            "temperature": (
                ("time", "level", "latitude", "longitude"),
                temp_all,
                {
                    "units": "K",
                    "long_name": "Air Temperature",
                },
            ),
            "u_wind": (
                ("time", "level", "latitude", "longitude"),
                u_wind_all,
                {
                    "units": "m s-1",
                    "long_name": "Zonal Wind",
                },
            ),
            "v_wind": (
                ("time", "level", "latitude", "longitude"),
                v_wind_all,
                {
                    "units": "m s-1",
                    "long_name": "Meridional Wind",
                },
            ),
            "ozone": (
                ("time", "level", "latitude", "longitude"),
                ozone_all,
                {
                    "units": "ppbv",
                    "long_name": "Ozone Concentration",
                },
            ),
        },
        coords={
            "time": times,
            "level": (
                "level",
                levels,
                {"units": "hPa", "long_name": "Pressure Level"},
            ),
            "latitude": (
                "latitude",
                lats,
                {"units": "degrees_north", "long_name": "Latitude"},
            ),
            "longitude": (
                "longitude",
                lons,
                {"units": "degrees_east", "long_name": "Longitude"},
            ),
        },
        attrs={
            "description": "Synthetic atmospheric and air composition data for plotting examples.",
            "history": f"Created on {pd.Timestamp.now()}.",
        },
    )
    return ds

if __name__ == "__main__":
    # Example of how to use the function
    dataset = create_dataset()
    print(dataset)
